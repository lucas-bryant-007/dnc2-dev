"""Shared hyper-rectangle plotting (matplotlib only -- no model/data deps).

Holds the proposal figures: the clean 3D box (``plot_box_3d``) and the task-axis
interference heatmap (``plot_cosine_heatmap``). Kept free of torch-model / dataset
imports so it can be unit-/smoke-tested on its own.
"""


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
        ax.scatter(p[sel, 0], p[sel, 1], p[sel, 2], s=7, alpha=0.30,
                   color=cmap(idx), edgecolors="none", label=combo_label(combo))

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
    ax.view_init(elev=18, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    style.maybe_title(ax, title)
    leg = ax.legend(loc="upper left", fontsize=10, markerscale=2.2,
                    framealpha=0.95, title="granular task")
    leg.get_title().set_fontsize(11)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
