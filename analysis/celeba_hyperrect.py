"""Multi-attribute / hyper-rectangle analysis on a frozen CelebA SSL backbone.

Runs on a *single* checkpoint (default epoch 1000) -- no retraining. It extracts
features once, carries many CelebA attribute labels in the same pass, then for
each attribute computes the task direction delta_t = mu_t^+ - mu_t^- and its
directional CDNV, measures pairwise "interference" cos(delta_s, delta_t)
(orthogonal task axes => hyper-rectangle structure), and renders a 3D box over
the three most-orthogonal attributes.

Example
-------
    # See what attribute columns the dataset actually exposes:
    python -u analysis/celeba_hyperrect.py --config configs/ijepa/celeba.yaml \
        --ckpt_dir <dir> --list_attributes

    # Full run on epoch 1000:
    python -u analysis/celeba_hyperrect.py \
        --config configs/ijepa/celeba.yaml \
        --ckpt_dir /home/.../ijepa-resnet50-celeba/converted_checkpoints \
        --device cuda:0 --epoch 1000 --tag twoview
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils import CelebACfg, CelebADataModule
from eval_utils import (
    find_checkpoint_files,
    load_model_from_checkpoint,
    extract_backbone_features,
    extract_features,
    set_seed,
    freeze_model,
)
from br.ssl_subspace import fit_ssl_subspace
from box_viz import plot_box_3d, plot_cosine_heatmap
import hyperrect as H
import metrics_io as mio

# Standard 40 CelebA attributes (used as the default request set; only those
# actually present in the loaded dataset are kept).
CELEBA_40 = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young",
]


def resolve_attributes(ds, requested):
    """Intersect requested attribute names with the dataset's columns."""
    available = list(getattr(ds, "column_names", []))
    if requested:
        missing = [a for a in requested if a not in available]
        if missing:
            raise SystemExit(
                f"Requested attributes not in dataset columns: {missing}\n"
                f"Available columns: {available}"
            )
        return list(requested)
    keep = [a for a in CELEBA_40 if a in available]
    if not keep:
        raise SystemExit(
            "None of the standard CelebA attributes were found as columns.\n"
            f"Available columns: {available}\n"
            "Pass --attributes <names...> explicitly."
        )
    return keep


def make_collate(image_key, tfms, attr_names):
    def collate(batch):
        imgs = torch.stack([tfms(ex[image_key].convert("RGB")) for ex in batch])
        attrs = torch.tensor(
            [[int(ex[a]) for a in attr_names] for ex in batch], dtype=torch.long
        )
        return imgs, attrs
    return collate


@torch.no_grad()
def extract_features_and_attrs(loader, backbone, device, max_samples=None):
    feats_list, attr_list = [], []
    seen = 0
    for imgs, attrs in tqdm(loader):
        imgs = imgs.to(device, non_blocking=True)
        feats = extract_backbone_features(backbone, imgs)
        feats = F.normalize(feats, dim=1)
        feats_list.append(feats.cpu())
        attr_list.append(attrs)
        seen += imgs.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    features = torch.cat(feats_list, dim=0)
    attr_matrix = torch.cat(attr_list, dim=0)
    if max_samples is not None:
        features = features[:max_samples]
        attr_matrix = attr_matrix[:max_samples]
    return features, attr_matrix


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))

    data_cfg = CelebACfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    data_cfg.batch_size = args.batch_size
    data_module = CelebADataModule(data_cfg)
    data_module.setup()

    ds = data_module.ds_train if args.split == "train" else data_module.ds_test

    if args.list_attributes:
        print("Dataset columns:", list(getattr(ds, "column_names", [])))
        return

    attr_names = resolve_attributes(ds, args.attributes)
    print(f"Using {len(attr_names)} attributes: {attr_names}")

    # Locate the requested epoch checkpoint.
    ckpt_files = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(
            f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}. "
            f"Found epochs: {sorted(k for k in ckpt_files if isinstance(k, int))}"
        )
    ckpt_path = ckpt_files[args.epoch]
    print(f"Loading checkpoint: {ckpt_path} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(args.device)
    freeze_model(model)

    collate = make_collate(data_cfg.image_key, data_module.test_tfms, attr_names)
    loader = DataLoader(
        ds, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True, collate_fn=collate,
    )

    features, attr_matrix = extract_features_and_attrs(
        loader, model.backbone, args.device, max_samples=args.max_samples
    )
    print(f"Extracted features {tuple(features.shape)}, attrs {tuple(attr_matrix.shape)}")

    first_stage_ssl_whitener = None
    # Optional: map into the whitened SSL subspace (paper-faithful geometry).
    # Thm 4.4's sqrt(B_t) hyper-rectangle and the one-scalar B(F) results hold
    # for centered+whitened F; for VICReg/Barlow the raw box is a parallelepiped
    # (App. A) until rewhitened, which is exactly what this does.
    if args.whiten:
        paired_loader = data_module.paired_train_dataloader()
        z1, z2, _ = extract_features(
            paired_loader, model.backbone, device=args.device, both_views=True)
        estimator = fit_ssl_subspace(
            z1, z2, rel_eig_threshold=args.rel_eig_threshold)
        first_stage_ssl_whitener = estimator.first_stage_whitener_provenance(
            fit_split="train",
            fit_population="full_training_latent_instance_population",
            view_marginal=(
                "equal_weight_empirical_mixture_of_two_augmented_views_per_instance"
            ),
            frozen_for_test=(args.split == "test"),
        )
        features = estimator.transform(features)  # -> psi (whitened SSL coords)
        print(f"Whitened to psi: {tuple(features.shape)} (k_eff={estimator.k_eff})")

    # Geometry on GPU if available (40 attributes x ~160k x D).
    feats_dev = features.to(args.device)
    rewhitening_record = None
    if args.whiten:
        # Rewhiten psi by its OWN covariance (Mahalanobis metric, App. A): the
        # SSL map isn't white on the eval distribution, so without this the
        # capture B can exceed 1 and the box is a skewed parallelepiped.
        feats_dev, rewhitener = H.rewhiten(
            feats_dev,
            return_transform=True,
        )
        rewhitening_record = {
            **rewhitener.metadata(),
            "scope": "same_population_fit_and_evaluation",
            "diagnostics": H.whitening_diagnostics(feats_dev).as_dict(),
        }
        print("Exactly re-whitened retained psi coordinates (Cov = I on fit data).")
    attrs_dev = attr_matrix.to(args.device)
    res = H.analyze(
        feats_dev, attrs_dev, attr_names,
        min_class_frac=args.min_class_frac,
        viz_triple=(args.viz_attrs if args.viz_attrs else None),
        compute_capture=args.whiten,
        min_capture=args.min_capture,
        cos_ceiling=args.cos_ceiling,
    )

    method = str(cfg.method.name)
    tag = (args.tag or "").strip()
    suffix = f"_{mio.slug(tag)}" if tag else ""
    stem = f"{mio.slug(method)}_celeba_epoch_{args.epoch}{suffix}"
    fig_dir = os.path.join(args.out_dir, "figures")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"\nMean off-diagonal |cos| (interference): {res['mean_abs_offdiag_cosine']}")
    if res["triple_names"]:
        mbyname = {m["name"]: m for m in res["metrics"]}
        axinfo = ", ".join(
            f"{n}(sqrtB={math.sqrt(max(mbyname[n].get('capture_B') or 0.0, 0.0)):.2f})"
            for n in res["triple_names"])
        print(f"Box triple: {res['triple_names']}  max pairwise |cos|={res['triple_max_abs_cos']:.3f}")
        print(f"  axes: {axinfo}")
    print("\nPer-attribute, sorted by capture (best first; * = ~balanced):")
    usable_sorted = sorted(
        [m for m in res["metrics"] if m.get("usable")],
        key=H.capture_proxy,
        reverse=True,
    )
    for m in usable_sorted:
        capB = m.get("capture_B")
        cap_str = (f"  B={capB:.3f} sqrtB={math.sqrt(max(capB,0.0)):.3f}"
                   if capB is not None else "")
        bal = " *" if 0.30 <= m["pos_frac"] <= 0.70 else ""
        print(f"  {m['name']:>22s}  dirCDNV={m['directional_cdnv']:.4f}{cap_str}  "
              f"gap={m['gap']:.4f}  pos_frac={m['pos_frac']:.3f}{bal}")

    # --- metrics export ---
    coords = res.pop("coords")
    granular_task = res.pop("granular_task")
    payload = {
        "method": method, "tag": tag or None, "epoch": args.epoch,
        "split": args.split, "config": args.config, "ckpt_path": ckpt_path,
        "n_samples": res["n_samples"], "feature_dim": res["feature_dim"],
        "whitened": bool(args.whiten),
        "first_stage_ssl_whitener": first_stage_ssl_whitener,
        "rewhitening": rewhitening_record,
        **{k: res[k] for k in (
            "attributes", "metrics", "cosine_matrix", "mean_abs_offdiag_cosine",
            "triple_names", "triple_max_abs_cos", "box", "predicted_box")},
    }
    json_path = mio.write_json(os.path.join(metrics_dir, f"hyperrect_{stem}.json"), payload)
    print(f"\nSaved JSON: {json_path}")

    attr_rows = [{
        "method": method, "epoch": args.epoch, "tag": tag, "attribute": m["name"],
        "n_pos": m.get("n_pos"), "n_neg": m.get("n_neg"), "pos_frac": m.get("pos_frac"),
        "gap": m.get("gap"), "directional_cdnv": m.get("directional_cdnv"),
        "cdnv": m.get("cdnv"), "capture_B": m.get("capture_B"),
        "sqrt_capture_B": (math.sqrt(m["capture_B"]) if m.get("capture_B") is not None else None),
        "usable": m.get("usable"),
    } for m in res["metrics"]]
    attr_csv = mio.write_table(
        os.path.join(metrics_dir, f"hyperrect_attrs_{stem}.csv"),
        ["method", "epoch", "tag", "attribute", "n_pos", "n_neg", "pos_frac",
         "gap", "directional_cdnv", "cdnv", "capture_B", "sqrt_capture_B", "usable"],
        attr_rows,
    )
    print(f"Saved attribute CSV: {attr_csv}")

    cos = res["cosine_matrix"]
    cos_rows = []
    for i, ni in enumerate(attr_names):
        for j, nj in enumerate(attr_names):
            if j <= i:
                continue
            cos_rows.append({
                "method": method, "epoch": args.epoch, "tag": tag,
                "attr_i": ni, "attr_j": nj,
                "cosine": cos[i][j], "abs_cosine": abs(cos[i][j]),
            })
    cos_csv = mio.write_table(
        os.path.join(metrics_dir, f"hyperrect_cosine_{stem}.csv"),
        ["method", "epoch", "tag", "attr_i", "attr_j", "cosine", "abs_cosine"],
        cos_rows,
    )
    print(f"Saved cosine CSV: {cos_csv}")

    # --- figures ---
    try:
        usable_idx = [i for i, m in enumerate(res["metrics"]) if m.get("usable")]
        if len(usable_idx) >= 2:
            sub = torch.tensor(res["cosine_matrix"]).abs()[usable_idx][:, usable_idx]
            sub_names = [attr_names[i] for i in usable_idx]
            heat_path = os.path.join(fig_dir, f"hyperrect_cosine_{stem}.png")
            plot_cosine_heatmap(sub.numpy(), sub_names, heat_path,
                                title=f"{method} CelebA epoch {args.epoch}: task-axis |cos|")
            print(f"Saved cosine heatmap: {heat_path}")
        if res["box"] is not None and coords is not None and granular_task is not None:
            box_path = os.path.join(fig_dir, f"hyperrect_box_{stem}.png")
            space = "whitened psi" if args.whiten else "raw features"
            plot_box_3d(coords.cpu(), res["box"], granular_task.cpu(),
                        res["triple_names"], box_path,
                        predicted_box=(res.get("predicted_box")
                                       if args.show_predicted_box else None),
                        title=f"{method} CelebA epoch {args.epoch} ({space}): "
                              f"{' / '.join(res['triple_names'])}")
            print(f"Saved 3D box: {box_path}")
    except Exception as e:  # noqa: BLE001 - never lose metrics over a plotting error
        print(f"WARNING: plotting failed ({type(e).__name__}: {e}); metrics were saved.")

    print("\nFinished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--ckpt_dir", "-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Feature-extraction batch size")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="Single checkpoint epoch to analyze")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--attributes", nargs="*", default=None,
                        help="Attribute columns to use (default: standard CelebA 40 present in the data)")
    parser.add_argument("--viz_attrs", nargs=3, default=None,
                        help="Three attribute names for the 3D box (default: auto-pick most orthogonal)")
    parser.add_argument("--min_class_frac", type=float, default=0.02,
                        help="Minority-class fraction below which an attribute is excluded from the box triple")
    parser.add_argument("--show_predicted_box", action="store_true",
                        help="Overlay the Thm 4.4 sqrt(B_t) predicted corners (validation "
                             "view); off by default for a single clean box")
    parser.add_argument("--min_capture", type=float, default=0.10,
                        help="Min per-attribute capture B for box-triple candidates")
    parser.add_argument("--cos_ceiling", type=float, default=0.40,
                        help="Max pairwise |cos| allowed for the auto-picked box triple "
                             "(best-captured triple under this ceiling is chosen)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of samples (for speed)")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--list_attributes", action="store_true",
                        help="Print dataset columns and exit")
    parser.add_argument("--whiten", action="store_true",
                        help="Map features into the whitened SSL subspace (two-view) "
                             "for the paper-faithful sqrt(B_t) hyper-rectangle")
    parser.add_argument("--rel_eig_threshold", type=float, default=1e-3,
                        help="Whitening: keep covariance directions >= rel*lambda_1")
    main(parser.parse_args())
