"""Shared hyper-rectangle plotting (matplotlib only -- no model/data deps).

Holds the proposal figures: the clean 3D box (``plot_box_3d``), the task-axis
interference heatmap (``plot_cosine_heatmap``), and a single animation frame
(``render_box_frame``) used to build the "cube assembling over training" GIF.
Kept free of torch-model / dataset imports so it can be smoke-tested on its own.
"""


def render_box_frame(pts, pts_task, centroids, triple_names, lim, arrow_len,
                     elev=20, azim=-55, epoch_label=None, point_size=10,
                     point_alpha=0.55, dpi=110):
    """Render one animation frame from explicit point/centroid data -> RGB ndarray.

    ``pts`` [M,3] sample coords, ``pts_task`` [M] granular-task index (0..7) for
    color, ``centroids`` dict {(b0,b1,b2): [x,y,z]}. ``lim`` and ``arrow_len`` are
    held fixed across the whole animation so the camera doesn't jump; only the box
    + swarm morph. Returns an (H,W,3) uint8 array (no file written).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    from br import style
    style.apply_style()

    fig = plt.figure(figsize=(7.5, 6.5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.tab10
    pts = np.asarray(pts)
    pts_task = np.asarray(pts_task)

    for idx in range(8):
        sel = np.where(pts_task == idx)[0]
        if sel.size == 0:
            continue
        combo = ((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)
        lbl = " ".join((("+" if b else "-") + n) for n, b in zip(triple_names, combo))
        ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], s=point_size,
                   alpha=point_alpha, color=cmap(idx), edgecolors="none",
                   depthshade=True, label=lbl)

    for combo, ctr in centroids.items():
        idx = combo[0] * 4 + combo[1] * 2 + combo[2]
        ax.scatter(ctr[0], ctr[1], ctr[2], s=210, color=cmap(idx),
                   edgecolor="black", linewidth=1.6, depthshade=False, zorder=5)
    for combo, ctr in centroids.items():
        for axis in range(3):
            nbr = list(combo); nbr[axis] ^= 1; nbr = tuple(nbr)
            if nbr in centroids and nbr > combo:
                q = centroids[nbr]
                ax.plot([ctr[0], q[0]], [ctr[1], q[1]], [ctr[2], q[2]],
                        color="black", linewidth=2.0, alpha=0.85, zorder=4)

    for k, vec in enumerate([(arrow_len, 0, 0), (0, arrow_len, 0), (0, 0, arrow_len)]):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color="black",
                  linewidth=2.0, arrow_length_ratio=0.12)
        ax.text(vec[0] * 1.16, vec[1] * 1.16, vec[2] * 1.16, triple_names[k],
                fontsize=12, fontweight="bold", ha="center", va="center")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white"); a.pane.set_alpha(1.0); a.pane.set_edgecolor("0.88")
    if epoch_label:
        ax.text2D(0.04, 0.93, epoch_label, transform=ax.transAxes, fontsize=15,
                  fontweight="bold", ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    leg = ax.legend(loc="upper right", fontsize=8, markerscale=1.6,
                    framealpha=0.9, title="granular task")
    leg.get_title().set_fontsize(9)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _interp_centroids(ca, cb, t):
    return {c: [(1 - t) * ca[c][d] + t * cb[c][d] for d in range(3)]
            for c in ca if c in cb}


def build_gif(keyframes, plot_idx, plot_task, names, out_path,
              hold=5, steps=6, fps=12, rot_speed=0.7, per_point=10,
              point_alpha=0.55, dpi=110, lim=None, arrow_len=None):
    """Morph the box across epochs and save a looping GIF (model-free).

    ``keyframes`` is a list of {epoch, coords[N,3], centroids} ordered by epoch;
    ``plot_idx`` selects the (fixed) samples to draw so each point morphs smoothly.
    ``lim``/``arrow_len`` are held constant so the camera doesn't jump.
    """
    import numpy as np
    from PIL import Image

    pts_by_epoch = [np.asarray(k["coords"])[plot_idx] for k in keyframes]
    if lim is None:
        allpts = np.concatenate(pts_by_epoch, axis=0)
        lim = float(max(1.0, np.percentile(np.abs(allpts), 99.0) * 1.15))
    if arrow_len is None:
        fc = np.array(list(keyframes[-1]["centroids"].values()))
        arrow_len = float(np.abs(fc).max()) if fc.size else 0.9

    frames, gi = [], 0

    def emit(pts, cents, label):
        nonlocal gi
        img = render_box_frame(pts, plot_task, cents, names, lim, arrow_len,
                               azim=-55 + rot_speed * gi, epoch_label=label,
                               point_size=per_point, point_alpha=point_alpha, dpi=dpi)
        frames.append(Image.fromarray(img)); gi += 1

    for i, kf in enumerate(keyframes):
        for _ in range(hold):
            emit(pts_by_epoch[i], kf["centroids"], f"epoch {kf['epoch']}")
        if i < len(keyframes) - 1 and steps > 0:
            nxt = keyframes[i + 1]
            for s in range(1, steps + 1):
                t = s / (steps + 1)
                pts_i = (1 - t) * pts_by_epoch[i] + t * pts_by_epoch[i + 1]
                emit(pts_i, _interp_centroids(kf["centroids"], nxt["centroids"], t),
                     f"epoch {kf['epoch']}→{nxt['epoch']}")
    for _ in range(fps):  # hold on the final clean cube
        emit(pts_by_epoch[-1], keyframes[-1]["centroids"],
             f"epoch {keyframes[-1]['epoch']}")

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"Saved GIF: {out_path}  ({len(frames)} frames @ {fps} fps, lim={lim:.2f})")
    return out_path


def plot_cosine_heatmap(cos_abs, names, save_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from br import style
    style.apply_style()

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * n), max(5, 0.4 * n)))
    im = ax.imshow(cos_abs, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=r"$|\cos(\delta_s,\delta_t)|$")
    style.maybe_title(ax, title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_box_3d(coords, box, granular_task, triple_names, save_path,
                predicted_box=None, per_task=500, title=None, zoom=1.5):
    """Clean 3D hyper-rectangle for the proposal: a swarm of samples colored by
    granular task clustered around each of the 8 centroids, a bold box through
    the centroids, and labeled arrows along the three task axes (orthogonality).
    Tick numbers are dropped; the predicted sqrt(B_t) overlay is opt-in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    from br import style
    style.apply_style()

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.tab10
    p = coords.numpy() if hasattr(coords, "numpy") else np.asarray(coords)
    g = granular_task.numpy() if hasattr(granular_task, "numpy") else np.asarray(granular_task)

    def combo_label(combo):
        return " ".join((("+" if b else "-") + n) for n, b in zip(triple_names, combo))

    # Random samples from each granular task, colored by task.
    for idx in range(8):
        combo = ((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)
        sel = np.where(g == idx)[0]
        if sel.size == 0:
            continue
        if sel.size > per_task:
            sel = np.random.choice(sel, per_task, replace=False)
        ax.scatter(p[sel, 0], p[sel, 1], p[sel, 2], s=10, alpha=0.55,
                   color=cmap(idx), edgecolors="none", depthshade=True,
                   label=combo_label(combo))

    # The 8 granular-task centroids (box corners) in matching colors.
    centers = {tuple(e["combo"]): e["center"] for e in box if e["center"] is not None}
    for combo, ctr in centers.items():
        idx = combo[0] * 4 + combo[1] * 2 + combo[2]
        ax.scatter(ctr[0], ctr[1], ctr[2], s=240, color=cmap(idx),
                   edgecolor="black", linewidth=1.6, depthshade=False, zorder=5)
    # Bold box: connect centroids differing in exactly one task bit.
    for combo, ctr in centers.items():
        for axis in range(3):
            nbr = list(combo); nbr[axis] ^= 1; nbr = tuple(nbr)
            if nbr in centers and nbr > combo:
                q = centers[nbr]
                ax.plot([ctr[0], q[0]], [ctr[1], q[1]], [ctr[2], q[2]],
                        color="black", linewidth=2.2, alpha=0.85, zorder=4)

    # Predicted Thm 4.4 corners (hollow diamonds + dashed wireframe).
    if predicted_box is not None:
        pcent = {tuple(e["combo"]): e["center"] for e in predicted_box
                 if e["center"] is not None}
        for combo, ctr in pcent.items():
            ax.scatter(ctr[0], ctr[1], ctr[2], s=120, marker="D",
                       facecolors="none", edgecolors="black", linewidth=1.4,
                       depthshade=False)
        for combo, ctr in pcent.items():
            for axis in range(3):
                nbr = list(combo); nbr[axis] ^= 1; nbr = tuple(nbr)
                if nbr in pcent and nbr > combo:
                    q = pcent[nbr]
                    ax.plot([ctr[0], q[0]], [ctr[1], q[1]], [ctr[2], q[2]],
                            color="black", linewidth=1.0, alpha=0.4, linestyle="--")

    # Labeled arrows along the three task axes (shows orthogonality).
    allc = np.array(list(centers.values())) if centers else np.array([[1.0, 1.0, 1.0]])
    cext = float(np.abs(allc).max())
    L = cext * 1.1
    for k, vec in enumerate([(L, 0, 0), (0, L, 0), (0, 0, L)]):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color="black",
                  linewidth=2.0, arrow_length_ratio=0.12)
        ax.text(vec[0] * 1.16, vec[1] * 1.16, vec[2] * 1.16, triple_names[k],
                fontsize=12, fontweight="bold", ha="center", va="center")

    # Zoom so the box fills the frame; drop distracting tick numbers.
    lim = max(0.8, cext * zoom)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=20, azim=-55)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Clean white background panes, no grid -- keeps focus on the box + swarms.
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor("0.88")
    style.maybe_title(ax, title)
    leg = ax.legend(loc="upper left", fontsize=10, markerscale=2.2,
                    framealpha=0.95, title="granular task")
    leg.get_title().set_fontsize(11)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
