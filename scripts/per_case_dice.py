import os

import csv

import nibabel as nib

import numpy as np



gt_dir = "holdout/labelsTs"

baseline_dir = "holdout_predictions/baseline"

cldice_dir = "holdout_predictions/cldice"



def dice(gt, pred, label):

    g = gt == label

    p = pred == label

    denom = g.sum() + p.sum()

    if denom == 0:

        return 1.0

    return 2.0 * np.logical_and(g, p).sum() / denom



rows = []



for fname in sorted(os.listdir(gt_dir)):

    if not fname.endswith(".nii.gz"):

        continue



    gt = nib.load(os.path.join(gt_dir, fname)).get_fdata()

    base = nib.load(os.path.join(baseline_dir, fname)).get_fdata()

    cl = nib.load(os.path.join(cldice_dir, fname)).get_fdata()



    bv = dice(gt, base, 1)

    bt = dice(gt, base, 2)

    cv = dice(gt, cl, 1)

    ct = dice(gt, cl, 2)



    rows.append({

        "case": fname.replace(".nii.gz", ""),

        "baseline_vessel": bv,

        "baseline_tumor": bt,

        "baseline_mean": (bv + bt) / 2,

        "cldice_vessel": cv,

        "cldice_tumor": ct,

        "cldice_mean": (cv + ct) / 2,

        "mean_change": ((cv + ct) - (bv + bt)) / 2

    })



os.makedirs("results/metrics", exist_ok=True)



out = "results/metrics/per_case_holdout_dice.csv"



with open(out, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=rows[0].keys())

    writer.writeheader()

    writer.writerows(rows)



print("\nWORST BASELINE MEAN DICE")

for r in sorted(rows, key=lambda x: x["baseline_mean"])[:10]:

    print(r["case"],

          f"vessel={r['baseline_vessel']:.4f}",

          f"tumor={r['baseline_tumor']:.4f}",

          f"mean={r['baseline_mean']:.4f}")



print("\nBEST BASELINE MEAN DICE")

for r in sorted(rows, key=lambda x: x["baseline_mean"], reverse=True)[:10]:

    print(r["case"],

          f"vessel={r['baseline_vessel']:.4f}",

          f"tumor={r['baseline_tumor']:.4f}",

          f"mean={r['baseline_mean']:.4f}")



print("\nBIGGEST clDice IMPROVEMENTS")

for r in sorted(rows, key=lambda x: x["mean_change"], reverse=True)[:10]:

    print(r["case"],

          f"baseline={r['baseline_mean']:.4f}",

          f"cldice={r['cldice_mean']:.4f}",

          f"change={r['mean_change']:+.4f}")



print("\nBIGGEST clDice DECREASES")

for r in sorted(rows, key=lambda x: x["mean_change"])[:10]:

    print(r["case"],

          f"baseline={r['baseline_mean']:.4f}",

          f"cldice={r['cldice_mean']:.4f}",

          f"change={r['mean_change']:+.4f}")



print(f"\nSaved: {out}")
