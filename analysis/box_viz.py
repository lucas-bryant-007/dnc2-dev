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

    # Drop points outside the (fixed) frame so the random-epoch blob doesn't
    # spill past the box; as training organizes them they migrate inside.
    inb = (np.abs(pts[:, 0]) <= lim) & (np.abs(pts[:, 1]) <= lim) & (np.abs(pts[:, 2]) <= lim)
    for idx in range(8):
        sel = np.where((pts_task == idx) & inb)[0]
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
              hold=2, steps=14, fps=12, rot_speed=0.0, per_point=10,
              point_alpha=0.55, dpi=110, lim=None, arrow_len=None,
              elev=20, azim=-55, min_steps=2):
    """Morph the box across epochs and save a looping GIF (model-free).

    ``keyframes`` is a list of {epoch, coords[N,3], centroids} ordered by epoch;
    ``plot_idx`` selects the (fixed) samples so each point morphs smoothly.

    The motion is *adaptively paced*: each epoch->epoch segment gets a number of
    interpolation frames proportional to how much the box moves (so the early,
    fast-organizing epochs get many smooth frames and the converged tail goes
    quickly), with smoothstep easing. The camera is fixed by default
    (``rot_speed=0``) and the frame fits the final cube so it stays put.
    """
    import os
    import numpy as np
    from PIL import Image

    pts_by_epoch = [np.asarray(k["coords"])[plot_idx] for k in keyframes]
    if lim is None:  # fit the converged cube; the early blob just clips into frame
        fp = pts_by_epoch[-1]
        lim = float(max(1.5, np.percentile(np.abs(fp), 97.0) * 1.25))
    if arrow_len is None:
        fc = np.array(list(keyframes[-1]["centroids"].values()))
        arrow_len = float(np.abs(fc).max()) if fc.size else 0.9

    # Per-segment frame budget ~ how far the centroids move (where the action is).
    dists = []
    for i in range(len(keyframes) - 1):
        ca, cb = keyframes[i]["centroids"], keyframes[i + 1]["centroids"]
        d = (np.mean([np.linalg.norm(np.array(cb[c]) - np.array(ca[c]))
                      for c in ca if c in cb]) if ca else 0.0)
        dists.append(d)
    maxd = max(dists) if dists and max(dists) > 0 else 1.0
    seg_steps = [int(round(min_steps + (steps - min_steps) * (d / maxd))) for d in dists]

    frames = []
    counter = {"i": 0}

    def emit(pts, cents, label):
        img = render_box_frame(pts, plot_task, cents, names, lim, arrow_len,
                               elev=elev, azim=azim + rot_speed * counter["i"],
                               epoch_label=label, point_size=per_point,
                               point_alpha=point_alpha, dpi=dpi)
        frames.append(Image.fromarray(img)); counter["i"] += 1

    def ease(t):  # smoothstep ease-in-out
        return t * t * (3.0 - 2.0 * t)

    for i, kf in enumerate(keyframes):
        h = hold + (fps // 2 if i == 0 else 0)  # linger on the random start
        for _ in range(h):
            emit(pts_by_epoch[i], kf["centroids"], f"epoch {kf['epoch']}")
        if i < len(keyframes) - 1:
            nxt = keyframes[i + 1]
            ns = max(1, seg_steps[i])
            for s in range(1, ns + 1):
                t = ease(s / (ns + 1))
                pts_i = (1 - t) * pts_by_epoch[i] + t * pts_by_epoch[i + 1]
                emit(pts_i, _interp_centroids(kf["centroids"], nxt["centroids"], t),
                     f"epoch {kf['epoch']}→{nxt['epoch']}")
    for _ in range(fps):  # hold on the final clean cube
        emit(pts_by_epoch[-1], keyframes[-1]["centroids"],
             f"epoch {keyframes[-1]['epoch']}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"Saved GIF: {out_path}  ({len(frames)} frames @ {fps} fps, lim={lim:.2f}, "
          f"seg_steps={seg_steps})")
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
                predicted_box=None, per_task=500, title=None, zoom=1.12,
                axis_labels=None, level_labels=None):
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

    fig = plt.figure(figsize=(7.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.tab10
    p = coords.numpy() if hasattr(coords, "numpy") else np.asarray(coords)
    g = granular_task.numpy() if hasattr(granular_task, "numpy") else np.asarray(granular_task)

    alabels = axis_labels if axis_labels is not None else triple_names

    def combo_label(combo):
        if level_labels is not None:
            return " · ".join(level_labels[k][b] for k, b in enumerate(combo))
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
                   rasterized=True, label=combo_label(combo))

    pred_color = "#d62728"
    has_pred = predicted_box is not None
    centers = {tuple(e["combo"]): e["center"] for e in box if e["center"] is not None}
    pcent = ({tuple(e["combo"]): e["center"] for e in predicted_box
              if e["center"] is not None} if has_pred else {})

    def _edges(cent, **kw):
        for combo, ctr in cent.items():
            for axis in range(3):
                nbr = list(combo); nbr[axis] ^= 1; nbr = tuple(nbr)
                if nbr in cent and nbr > combo:
                    q = cent[nbr]
                    ax.plot([ctr[0], q[0]], [ctr[1], q[1]], [ctr[2], q[2]], **kw)

    # Observed box first, then the predicted sqrt(B_t) box (red dashed) on top so
    # the two stay distinguishable even when they nearly coincide (high B). Thin
    # the observed edges + frame the centroids with open red diamonds so the
    # prediction reads clearly instead of hiding under the bold black box.
    obs_lw = 1.6 if has_pred else 2.2
    _edges(centers, color="black", linewidth=obs_lw, alpha=0.85)
    if has_pred:
        _edges(pcent, color=pred_color, linewidth=1.7, alpha=0.95, linestyle=(0, (5, 4)))

    cs = 165 if has_pred else 240
    for combo, ctr in centers.items():
        idx = combo[0] * 4 + combo[1] * 2 + combo[2]
        ax.scatter(ctr[0], ctr[1], ctr[2], s=cs, color=cmap(idx),
                   edgecolor="black", linewidth=1.4, depthshade=False)
    if has_pred:  # open red diamonds framing each observed centroid
        for combo, ctr in pcent.items():
            ax.scatter(ctr[0], ctr[1], ctr[2], s=310, marker="D",
                       facecolors="none", edgecolors=pred_color, linewidth=2.0,
                       depthshade=False)

    # Labeled arrows along the three task axes (shows orthogonality). Labels are
    # pushed beyond the arrow tips and aligned to grow *outward* (for this fixed
    # view) so they don't sit on top of the arrows.
    allc = np.array(list(centers.values())) if centers else np.array([[1.0, 1.0, 1.0]])
    cext = float(np.abs(allc).max())
    L = cext * 0.92
    aligns = [("left", "top"), ("left", "bottom"), ("center", "bottom")]
    for k, vec in enumerate([(L, 0, 0), (0, L, 0), (0, 0, L)]):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color="black",
                  linewidth=1.8, arrow_length_ratio=0.1)
        ha, va = aligns[k]
        ax.text(vec[0] * 1.14, vec[1] * 1.14, vec[2] * 1.14, alabels[k],
                fontsize=12, fontweight="bold", ha=ha, va=va)

    # Zoom so the box fills the frame; drop the axis frame entirely so the figure
    # crops tight to the box (no empty 3D floor pane / whitespace).
    lim = max(0.8, cext * zoom)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((1, 1, 1))  # equal aspect (undistorted cube)
    ax.set_axis_off()             # drop the frame -> tight crop, no floor whitespace
    style.maybe_title(ax, title)
    # Compact color key (granular tasks): two columns, small, tucked low-left.
    gt_handles, gt_labels = ax.get_legend_handles_labels()  # the 8 swarm scatters
    leg1 = ax.legend(gt_handles, gt_labels, loc="lower center", ncol=4, fontsize=8.5,
                     markerscale=1.3, framealpha=0.9, handletextpad=0.35,
                     columnspacing=0.9, labelspacing=0.3, borderpad=0.35,
                     borderaxespad=0.0, bbox_to_anchor=(0.5, 0.0))
    ax.add_artist(leg1)
    if has_pred:
        from matplotlib.lines import Line2D
        box_handles = [Line2D([0], [0], color="black", lw=1.8),
                       Line2D([0], [0], color=pred_color, lw=1.8, linestyle=(0, (5, 4)))]
        leg2 = ax.legend(box_handles, ["observed", r"predicted $\sqrt{B_t}$"],
                         loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for pth in (save_path if isinstance(save_path, (list, tuple)) else [save_path]):
        fig.savefig(pth, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
