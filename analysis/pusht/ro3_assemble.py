"""Assemble the results.json consumed by ro3_figure.py.

Panel (a) factors come from the main run's obs-vs-cap file; panel (b) horizon
sweep is gathered from per-horizon obs-vs-cap files named ..._H<h>.json, reading
one designated factor out of each.

    python analysis/pusht/ro3_assemble.py \
        --factors metrics/ro3_obs_r4.json \
        --sweep_glob 'metrics/ro3_obs_H*.json' \
        --sweep_factor 'T orientation' --out metrics/ro3_results.json
"""
import argparse
import glob
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="metrics/ro3_obs_r4.json")
    ap.add_argument("--sweep_glob", default="metrics/ro3_obs_H*.json")
    ap.add_argument("--sweep_factor", default="T orientation")
    ap.add_argument("--main_H", type=int, default=48,
                    help="horizon of the --factors run; added as a sweep point so "
                         "panel (a) is the endpoint of panel (b)")
    ap.add_argument("--out", default="metrics/ro3_results.json")
    args = ap.parse_args()

    fac = json.load(open(args.factors))

    rows = []
    for p in sorted(glob.glob(args.sweep_glob)):
        m = re.search(r"_H(\d+)", p)
        if not m:
            continue
        d = json.load(open(p))
        fr = next((f for f in d["factors"] if f["name"] == args.sweep_factor), None)
        if fr is not None:
            rows.append((int(m.group(1)), fr))
    # add the main run (panel a) as its own sweep point at main_H
    fr_main = next((f for f in fac["factors"] if f["name"] == args.sweep_factor), None)
    if fr_main is not None and args.main_H not in [h for h, _ in rows]:
        rows.append((args.main_H, fr_main))
    rows.sort()

    sweep = dict(
        factor=args.sweep_factor,
        H=[h for h, _ in rows],
        obs=[fr["obs"] for _, fr in rows],
        obs_sd=[fr.get("obs_sd", fr.get("obs_se", 0.0)) for _, fr in rows],
        cap=[fr["cap"] for _, fr in rows],
        cap_sd=[fr.get("cap_sd", fr.get("cap_se", 0.0)) for _, fr in rows])

    out = dict(factors=fac["factors"], sweep=sweep,
               n_seeds=fac.get("n_seeds", 3), bottleneck_r=fac.get("bottleneck_r", 4))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}: {len(fac['factors'])} factors, "
          f"{len(rows)} horizon points ({args.sweep_factor})")


if __name__ == "__main__":
    main()
