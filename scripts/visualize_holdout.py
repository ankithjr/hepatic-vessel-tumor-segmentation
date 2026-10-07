import os

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt



IMAGE_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout/imagesTs"

GT_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout/labelsTs"

BASELINE_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout_predictions/baseline"

CLDICE_DIR = "/mnt/vstor/courses/csds463/arj94/project1/holdout_predictions/cldice"

OUT_DIR = "/mnt/vstor/courses/csds463/arj94/project1/results/figures"



os.makedirs(OUT_DIR, exist_ok=True)



cases = [

    "hepaticvessel_002",

    "hepaticvessel_008",

    "hepaticvessel_019",

]



for case in cases:



    image_path = os.path.join(

        IMAGE_DIR,

        case + "_0000.nii.gz"

    )



    gt_path = os.path.join(

        GT_DIR,

        case + ".nii.gz"

    )



    baseline_path = os.path.join(

        BASELINE_DIR,

        case + ".nii.gz"

    )



    cldice_path = os.path.join(

        CLDICE_DIR,

        case + ".nii.gz"

    )



    image = nib.load(image_path).get_fdata()

    gt = nib.load(gt_path).get_fdata()

    baseline = nib.load(baseline_path).get_fdata()

    cldice = nib.load(cldice_path).get_fdata()



    # Choose slice with most foreground anatomy

    foreground_per_slice = np.sum(gt > 0, axis=(0, 1))

    slice_idx = np.argmax(foreground_per_slice)



    image_slice = image[:, :, slice_idx]

    gt_slice = gt[:, :, slice_idx]

    baseline_slice = baseline[:, :, slice_idx]

    cldice_slice = cldice[:, :, slice_idx]



    fig, axes = plt.subplots(

        1,

        4,

        figsize=(18, 5)

    )



    axes[0].imshow(

        image_slice.T,

        cmap="gray",

        origin="lower"

    )

    axes[0].set_title("CT")



    axes[1].imshow(

        image_slice.T,

        cmap="gray",

        origin="lower"

    )

    axes[1].imshow(

        gt_slice.T,

        alpha=0.5,

        origin="lower"

    )

    axes[1].set_title("Ground Truth")



    axes[2].imshow(

        image_slice.T,

        cmap="gray",

        origin="lower"

    )

    axes[2].imshow(

        baseline_slice.T,

        alpha=0.5,

        origin="lower"

    )

    axes[2].set_title("Baseline nnU-Net")



    axes[3].imshow(

        image_slice.T,

        cmap="gray",

        origin="lower"

    )

    axes[3].imshow(

        cldice_slice.T,

        alpha=0.5,

        origin="lower"

    )

    axes[3].set_title("nnU-Net + clDice")



    for ax in axes:

        ax.axis("off")



    plt.suptitle(

        f"{case} — Slice {slice_idx}"

    )



    plt.tight_layout()



    output_path = os.path.join(

        OUT_DIR,

        case + "_baseline_vs_cldice.png"

    )



    plt.savefig(

        output_path,

        dpi=200,

        bbox_inches="tight"

    )



    plt.close()



    print("Saved:", output_path)
