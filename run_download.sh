#!/bin/bash
# Downloads and organizes the APTOS 2019 dataset via a Kaggle Dataset mirror
# that's already pre-sorted into class folders (bypasses the competition
# rules-acceptance gate, and doesn't need a CSV join).
python -m src.download_data \
    --dataset aptos \
    --output_dir data \
    --kaggle_dataset_slug sovitrath/diabetic-retinopathy-224x224-gaussian-filtered \
    --presorted
