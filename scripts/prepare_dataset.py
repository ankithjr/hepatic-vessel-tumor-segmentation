from pathlib import Path

import random

import shutil

import json



source = Path("/mnt/vstor/courses/csds463/arj94/project1/Task08_HepaticVessel")



nnunet_dataset = Path(

    "/mnt/vstor/courses/csds463/arj94/nnUNet_raw/Dataset008_HepaticVessel"

)



holdout = Path(

    "/mnt/vstor/courses/csds463/arj94/project1/holdout"

)



imagesTr_out = nnunet_dataset / "imagesTr"

labelsTr_out = nnunet_dataset / "labelsTr"



holdout_images = holdout / "imagesTs"

holdout_labels = holdout / "labelsTs"



for folder in [

    imagesTr_out,

    labelsTr_out,

    holdout_images,

    holdout_labels,

]:

    folder.mkdir(parents=True, exist_ok=True)



images = sorted((source / "imagesTr").glob("*.nii.gz"))



cases = []



for image in images:

    case_id = image.name.replace(".nii.gz", "")

    label = source / "labelsTr" / f"{case_id}.nii.gz"



    if not label.exists():

        raise FileNotFoundError(f"Missing label for {case_id}")



    cases.append(case_id)



print(f"Found {len(cases)} labeled cases.")



random.seed(42)

random.shuffle(cases)



n_holdout = round(len(cases) * 0.20)



holdout_cases = sorted(cases[:n_holdout])

train_cases = sorted(cases[n_holdout:])



print(f"Training cases: {len(train_cases)}")

print(f"Holdout cases: {len(holdout_cases)}")



for case_id in train_cases:

    src_image = source / "imagesTr" / f"{case_id}.nii.gz"

    src_label = source / "labelsTr" / f"{case_id}.nii.gz"



    dst_image = imagesTr_out / f"{case_id}_0000.nii.gz"

    dst_label = labelsTr_out / f"{case_id}.nii.gz"



    shutil.copy2(src_image, dst_image)

    shutil.copy2(src_label, dst_label)



for case_id in holdout_cases:

    src_image = source / "imagesTr" / f"{case_id}.nii.gz"

    src_label = source / "labelsTr" / f"{case_id}.nii.gz"



    dst_image = holdout_images / f"{case_id}_0000.nii.gz"

    dst_label = holdout_labels / f"{case_id}.nii.gz"



    shutil.copy2(src_image, dst_image)

    shutil.copy2(src_label, dst_label)



dataset_json = {

    "channel_names": {

        "0": "CT"

    },

    "labels": {

        "background": 0,

        "Vessel": 1,

        "Tumour": 2

    },

    "numTraining": len(train_cases),

    "file_ending": ".nii.gz"

}



with open(nnunet_dataset / "dataset.json", "w") as f:

    json.dump(dataset_json, f, indent=4)



split_json = {

    "seed": 42,

    "training_cases": train_cases,

    "holdout_cases": holdout_cases

}



with open(holdout / "split.json", "w") as f:

    json.dump(split_json, f, indent=4)



print("\nDataset preparation complete.")

print(f"nnU-Net dataset: {nnunet_dataset}")

print(f"Holdout dataset: {holdout}")

