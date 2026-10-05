import os

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt



IMAGE_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout/imagesTs"

GT_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout/labelsTs"

PRED_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout_predictions/baseline"

OUT_DIR = "/mnt/vstor/courses/csds463/arj94/project1/results/figures"



os.makedirs(OUT_DIR, exist_ok=True)



cases = [

    "hepaticvessel_002",

    "hepaticvessel_008",

    "hepaticvessel_019",

]



for case in cases:

    image_path = os.path.join(IMAGE_DIR, case + "_0000.nii.gz")

    gt_path = os.path.join(GT_DIR, case + ".nii.gz")

    pred_path = os.path.join(PRED_DIR, case + ".nii.gz")



    image = nib.load(image_path).get_fdata()

    gt = nib.load(gt_path).get_fdata()

    pred = nib.load(pred_path).get_fdata()



    # Pick slice with largest amount of foreground anatomy in ground truth

    foreground_per_slice = np.sum(gt > 0, axis=(0, 1))

    slice_idx = int(np.argmax(foreground_per_slice))



    img_slice = image[:, :, slice_idx]

    gt_slice = gt[:, :, slice_idx]

    pred_slice = pred[:, :, slice_idx]



    fig, axes = plt.subplots(1, 3, figsize=(15, 5))



    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")

    axes[0].set_title("CT")

    axes[0].axis("off")



    axes[1].imshow(img_slice.T, cmap="gray", origin="lower")

    axes[1].imshow(

        np.ma.masked_where(gt_slice.T == 0, gt_slice.T),

        alpha=0.5,

        origin="lower"

    )

    axes[1].set_title("Ground Truth")

    axes[1].axis("off")



    axes[2].imshow(img_slice.T, cmap="gray", origin="lower")

    axes[2].imshow(

        np.ma.masked_where(pred_slice.T == 0, pred_slice.T),

        alpha=0.5,

        origin="lower"

    )

    axes[2].set_title("Prediction")

    axes[2].axis("off")



    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, case + "_comparison.png")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close()



    print(f"Saved {out_path}")

