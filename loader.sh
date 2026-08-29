#!/bin/bash

: '

PARA png:

BUCKET="gs://new_rgb_dataset"
LOCAL_ROOT="./datasets/dataset_png"

PARA TIFF:

BUCKET="gs://bucket_six_bands"
LOCAL_ROOT="./datasets/dataset_tiff"
'

echo "Empezando con script"

BUCKET="gs://new_rgb_dataset"
LOCAL_ROOT="./datasets/dataset_png"

folders=(
    "train/fire"
    "train/no_fire"
    "validation/fire"
    "validation/no_fire"
    "test/fire"
    "test/no_fire"
)

for folder in "${folders[@]}"; do
    echo "Downloading first 10 files from $folder..."

    local_dir="$LOCAL_ROOT/$folder"
    mkdir -p "$local_dir"

    mapfile -t files < <(
        gcloud storage ls "$BUCKET/dataset/$folder/*" | head -n 10
    )

    if [ ${#files[@]} -eq 0 ]; then
        echo "  No files found."
        continue
    fi

    gcloud storage cp "${files[@]}" "$local_dir/"

    echo "  Downloaded ${#files[@]} files."
done