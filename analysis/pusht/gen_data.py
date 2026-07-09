"""RO3 step 1: generate counterfactual Push-T futures.

For each sampled demo state we execute SIX candidate H-step action sequences
from the SAME initial state: the demonstrated sequence, four spatially shifted
copies, and one hold-position sequence. We record the initial
image X_t, the actions, the final image X_{t+H}, and ground-truth future
factors (final block pose, displacement, contact proxy, coverage c_t / c_{t+H},
goal progress).

Demos come from the diffusion_policy replay buffer
(pusht_cchi_v7_replay.zarr: data/state [N,5] = agent_xy + block_xy + block_angle,
data/action [N,2] = agent target position, meta/episode_ends). The simulator is
lerobot's pip-installable port of the same environment (gym-pusht).

Setup (once):
    pip install gymnasium gym-pusht zarr shapely pygame pymunk
    wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    unzip pusht.zip   # -> pusht/pusht_cchi_v7_replay.zarr

Smoke test first (CPU, ~1 min), then full run:
    python -u analysis/pusht/gen_data.py --zarr pusht/pusht_cchi_v7_replay.zarr \
        --n_states 8 --out data/pusht_cf_smoke.npz
    python -u analysis/pusht/gen_data.py --zarr pusht/pusht_cchi_v7_replay.zarr \
        --n_states 3000 --workers 16 --out data/pusht_cf.npz
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame rendering

import multiprocessing as mp

import numpy as np

# Defaults; overridable via --horizon / --shift. Longer H and larger, more
# diverse shifts make the six-plus candidate futures actually separate (bigger
# best-vs-worst coverage spread), so goal progress and regret are not dominated
# by near-ties.
H = 48              # control steps per rollout (10 Hz -> 4.8 s)
SHIFT = 48.0        # px offset for the shifted candidates (space is 512x512)
N_CAND = 8
_ENV = None         # per-worker singleton


def make_env():
    import gymnasium as gym
    import gym_pusht  # noqa: F401 (registers gym_pusht/PushT-v0)
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array")


# ---- exact (unclipped) goal coverage from env internals -------------------
def _to_shapely(body, shapes):
    import shapely.geometry as sg
    from shapely.ops import unary_union
    polys = []
    for shape in shapes:
        if hasattr(shape, "get_vertices"):
            verts = [tuple(body.local_to_world(v)) for v in shape.get_vertices()]
            polys.append(sg.Polygon(verts))
    return unary_union(polys)


def coverage(env):
    """intersection(block, goal) / area(goal); falls back to None if the
    gym-pusht internals ever change attribute names."""
    try:
        import pymunk
        u = env.unwrapped
        block_geom = _to_shapely(u.block, u.block.shapes)
        gb = pymunk.Body(1, float("inf"))
        gb.position = tuple(np.asarray(u.goal_pose, dtype=float)[:2])
        gb.angle = float(np.asarray(u.goal_pose, dtype=float)[2])
        goal_geom = _to_shapely(gb, u.block.shapes)
        return float(block_geom.intersection(goal_geom).area / goal_geom.area)
    except Exception:
        return None


def reset_to(env, state):
    obs, info = env.reset(options={"reset_to_state": np.asarray(state, float)})
    return obs, info


def candidates(demo_actions, agent_pos):
    """(N_CAND, H, 2): demo, 6 shifted copies (4 axis + 2 diagonal), hold."""
    offs = [(SHIFT, 0), (-SHIFT, 0), (0, SHIFT), (0, -SHIFT),
            (SHIFT, SHIFT), (-SHIFT, -SHIFT)]
    cands = [demo_actions]
    for dx, dy in offs:
        cands.append(np.clip(demo_actions + np.array([dx, dy]), 0.0, 512.0))
    cands.append(np.tile(np.asarray(agent_pos, float), (H, 1)))
    return np.stack(cands).astype(np.float32)


def _init_worker():
    global _ENV
    _ENV = make_env()


def rollout_state(job):
    """One sampled state -> X_t, six rollouts, factors."""
    state, demo_actions = job
    env = _ENV if _ENV is not None else make_env()
    out = dict(xf=[], cf=[], pose_f=[], disp=[], contact=[])

    obs, _ = reset_to(env, state)
    xt = obs["pixels"].astype(np.uint8)
    c0 = coverage(env)
    block0 = np.asarray(state, float)[2:4]

    for a_seq in candidates(demo_actions, state[:2]):
        reset_to(env, state)
        rew = 0.0
        for a in a_seq:
            obs, rew, term, trunc, info = env.step(a.astype(np.float32))
            if term or trunc:
                break
        cf = coverage(env)
        if cf is None:  # fallback: reward = clip(cov/thresh,0,1)
            cf = float(rew) * float(getattr(env.unwrapped,
                                            "success_threshold", 0.95))
        u = env.unwrapped
        pose = np.array([u.block.position[0], u.block.position[1],
                         u.block.angle], dtype=np.float32)
        disp = float(np.linalg.norm(pose[:2] - block0))
        out["xf"].append(obs["pixels"].astype(np.uint8))
        out["cf"].append(cf)
        out["pose_f"].append(pose)
        out["disp"].append(disp)
        out["contact"].append(disp > 1.0)  # block moved => contact happened

    if c0 is None:
        c0 = 0.0
    return (xt, np.float32(c0), np.stack(out["xf"]),
            np.asarray(out["cf"], np.float32),
            np.stack(out["pose_f"]), np.asarray(out["disp"], np.float32),
            np.asarray(out["contact"], bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True,
                    help="path to pusht_cchi_v7_replay.zarr")
    ap.add_argument("--n_states", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=H,
                    help="control steps per rollout (default 48)")
    ap.add_argument("--shift", type=float, default=SHIFT,
                    help="px offset for shifted candidates (default 48)")
    ap.add_argument("--out", default="data/pusht_cf.npz")
    args = ap.parse_args()

    # Override the module-level knobs before the (forked) workers read them.
    global H, SHIFT
    H, SHIFT = args.horizon, args.shift

    import zarr
    root = zarr.open(args.zarr, mode="r")
    states = np.asarray(root["data/state"])
    actions = np.asarray(root["data/action"])
    ep_ends = np.asarray(root["meta/episode_ends"])
    ep_starts = np.concatenate([[0], ep_ends[:-1]])

    rng = np.random.default_rng(args.seed)
    # sample (episode, t) pairs uniformly with the whole horizon in-episode
    pool = []
    for ep, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        for t in range(s, e - H):
            pool.append((ep, t))
    pool = np.asarray(pool)
    pick = pool[rng.choice(len(pool), size=min(args.n_states, len(pool)),
                           replace=False)]
    jobs = [(states[t], actions[t:t + H].astype(np.float32)) for _, t in pick]
    print(f"{len(jobs)} states x {N_CAND} candidates x {H} steps "
          f"({len(ep_ends)} episodes)")

    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker) as p:
            results = p.map(rollout_state, jobs, chunksize=4)
    else:
        _init_worker()
        results = []
        for i, j in enumerate(jobs):
            results.append(rollout_state(j))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    xt = np.stack([r[0] for r in results])
    c0 = np.asarray([r[1] for r in results], np.float32)
    xf = np.stack([r[2] for r in results])
    cf = np.stack([r[3] for r in results])
    pose_f = np.stack([r[4] for r in results])
    disp = np.stack([r[5] for r in results])
    contact = np.stack([r[6] for r in results])
    acts = np.stack([candidates(a, s[:2]) for s, a in jobs])
    progress = cf - c0[:, None]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, x_t=xt, x_f=xf, actions=acts, c_t=c0, c_f=cf,
        progress=progress, pose_f=pose_f, displacement=disp, contact=contact,
        episode=pick[:, 0].astype(np.int32), t=pick[:, 1].astype(np.int32),
        horizon=np.int32(H), shift=np.float32(SHIFT))
    print("saved", args.out,
          f"| coverage c_t mean {c0.mean():.3f} | progress "
          f"[{progress.min():.3f},{progress.max():.3f}] "
          f"| best-candidate spread "
          f"{(cf.max(1) - cf.min(1)).mean():.3f}")


if __name__ == "__main__":
    main()
