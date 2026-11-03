"""
Download and prepare the APTOS 2019 or EyePACS diabetic retinopathy
datasets from Kaggle, and lay them out as:

    data/train/<class_id>/*.png
    data/val/<class_id>/*.png

where class_id in {0,1,2,3,4} =
    0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR

Requires the Kaggle API (`pip install kaggle`) and a Kaggle API token
(kaggle.json) placed at ~/.kaggle/kaggle.json — see:
https://github.com/Kaggle/kaggle-api#api-credentials

Usage:
    # APTOS 2019 (Kaggle competition, ~10k images, id_code/diagnosis in train.csv)
    python -m src.download_data --dataset aptos --output_dir data

    # EyePACS (Kaggle competition, ~35k images, larger + noisier)
    python -m src.download_data --dataset eyepacs --output_dir data

    # Getting a 403 Forbidden from the competition endpoint (rules-acceptance
    # not registering, account restrictions, etc.)? Skip the competition
    # entirely and pull the same data from a public Kaggle Dataset mirror
    # instead — datasets don't require accepting competition rules:
    python -m src.download_data --dataset aptos --output_dir data \
        --kaggle_dataset_slug <owner>/<dataset-name>
    # Find a mirror by searching "APTOS 2019" under kaggle.com/datasets.

    # Already have the raw Kaggle files locally? Skip the download and just
    # organize them into class folders + a train/val split:
    python -m src.download_data --dataset aptos --output_dir data \
        --skip_download --raw_dir data/raw_aptos
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    0: "0_No_DR",
    1: "1_Mild",
    2: "2_Moderate",
    3: "3_Severe",
    4: "4_Proliferative_DR",
}

DATASET_CONFIG = {
    "aptos": {
        "kaggle_kind": "competition",
        "kaggle_ref": "aptos2019-blindness-detection",
        "labels_csv": "train.csv",
        "image_col": "id_code",
        "label_col": "diagnosis",
        "image_dir": "train_images",
        "image_ext": ".png",
    },
    "eyepacs": {
        "kaggle_kind": "competition",
        "kaggle_ref": "diabetic-retinopathy-detection",
        "labels_csv": "trainLabels.csv",
        "image_col": "image",
        "label_col": "level",
        "image_dir": "train",
        "image_ext": ".jpeg",
    },
}


def check_kaggle_cli():
    if shutil.which("kaggle") is None:
        sys.exit(
            "The `kaggle` CLI isn't installed / on PATH.\n"
            "Run: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json "
            "(from https://www.kaggle.com/settings -> Create New API Token)."
        )


def run_kaggle_command(cmd, context_hint=""):
    """Run a kaggle CLI command, surfacing a clear, actionable message on failure
    instead of a raw traceback."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        return

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    combined = f"{stdout}\n{stderr}".strip()

    if "403" in combined or "Forbidden" in combined:
        sys.exit(
            "Kaggle returned 403 Forbidden.\n"
            f"{context_hint}\n"
            "Most common causes:\n"
            "  1. You haven't accepted the competition rules — visit the competition's\n"
            "     /rules page on kaggle.com and click Accept (not just the homepage).\n"
            "  2. Your kaggle.json was generated before completing phone verification —\n"
            "     regenerate it (Settings -> API -> Create Legacy API Key) and replace\n"
            "     ~/.kaggle/kaggle.json.\n"
            "  3. This dataset/competition requires it entirely, but a public Dataset\n"
            "     mirror (not gated by competition rules) may exist — search kaggle.com/datasets\n"
            "     and rerun with --kaggle_dataset_slug <owner>/<name>.\n"
            f"\nRaw Kaggle output:\n{combined}"
        )
    elif "401" in combined or "Unauthorized" in combined:
        sys.exit(
            "Kaggle returned 401 Unauthorized — your credentials aren't valid.\n"
            "Regenerate kaggle.json at kaggle.com/settings/api ('Create Legacy API Key') "
            "and place it at ~/.kaggle/kaggle.json (chmod 600).\n"
            f"\nRaw Kaggle output:\n{combined}"
        )
    else:
        sys.exit(f"Kaggle command failed (exit code {result.returncode}).\n"
                  f"Raw Kaggle output:\n{combined}")


def download_dataset(cfg, raw_dir: Path):
    check_kaggle_cli()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {cfg['kaggle_ref']} from Kaggle "
          f"(you must have joined/accepted the competition rules on kaggle.com first)...")
    cmd = ["kaggle", "competitions", "download", "-c", cfg["kaggle_ref"], "-p", str(raw_dir)]
    run_kaggle_command(
        cmd,
        context_hint=f"Competition: {cfg['kaggle_ref']} "
                      f"(kaggle.com/c/{cfg['kaggle_ref']}/rules)",
    )
    _extract_zips(raw_dir, preferred_name=f"{cfg['kaggle_ref']}.zip")
    return raw_dir


def download_dataset_mirror(dataset_slug: str, raw_dir: Path):
    """Download from a Kaggle Dataset (not Competition) — no rules-acceptance gate."""
    check_kaggle_cli()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{dataset_slug}' from Kaggle (bypassing competition rules gate)...")
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir), "--unzip"]
    run_kaggle_command(
        cmd,
        context_hint=f"Dataset: {dataset_slug} (kaggle.com/datasets/{dataset_slug})",
    )
    # --unzip already extracts, but handle any leftover/nested zips just in case
    _extract_zips(raw_dir)
    return raw_dir


CLASS_KEYWORDS = {
    0: ["no_dr", "no dr", "nodr", "0_no", "0 -", "0-no", "healthy"],
    1: ["mild"],
    2: ["moderate"],
    3: ["severe"],
    4: ["proliferat"],
}


def _find_class_dir(raw_dir: Path, class_id: int):
    """Locate the directory for a given class by fuzzy name match, preferring
    the directory that actually contains image files (not just a labeled
    parent folder)."""
    keywords = CLASS_KEYWORDS[class_id]
    candidates = [
        d for d in raw_dir.rglob("*")
        if d.is_dir() and any(kw in d.name.lower() for kw in keywords)
    ]
    image_exts = {".png", ".jpg", ".jpeg"}
    for d in sorted(candidates, key=lambda p: len(str(p))):  # shallowest first
        if any(f.suffix.lower() in image_exts for f in d.iterdir() if f.is_file()):
            return d
    return candidates[0] if candidates else None


def organize_presorted_dataset(raw_dir: Path, output_dir: Path, val_split=0.15, seed=42):
    """For mirrors that are ALREADY split into one folder per class (no CSV
    join needed) — just re-shuffle each class's files into a train/val split."""
    import random
    rng = random.Random(seed)
    image_exts = {".png", ".jpg", ".jpeg"}

    class_dirs = {}
    for class_id in CLASS_NAMES:
        found = _find_class_dir(raw_dir, class_id)
        if found is None:
            sys.exit(
                f"Could not find a folder for class {class_id} ({CLASS_NAMES[class_id]}) "
                f"under {raw_dir}.\nRun `find {raw_dir} -maxdepth 4 -type d` to see what's "
                f"actually there, and let me know the folder names so I can adjust the matching."
            )
        class_dirs[class_id] = found

    print("Matched class folders:")
    for cid, d in class_dirs.items():
        print(f"  {CLASS_NAMES[cid]}: {d}")

    for split_name in ["train", "val"]:
        for cid in CLASS_NAMES:
            (output_dir / split_name / CLASS_NAMES[cid]).mkdir(parents=True, exist_ok=True)

    total_train, total_val = 0, 0
    for cid, src_dir in class_dirs.items():
        files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_split)) if files else 0
        val_files, train_files = files[:n_val], files[n_val:]

        for f in train_files:
            shutil.copyfile(f, output_dir / "train" / CLASS_NAMES[cid] / f.name)
        for f in val_files:
            shutil.copyfile(f, output_dir / "val" / CLASS_NAMES[cid] / f.name)

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"  {CLASS_NAMES[cid]}: {len(train_files)} train / {len(val_files)} val")

    print(f"\nTotal: {total_train} train, {total_val} val images organized.")


def _extract_zips(raw_dir: Path, preferred_name: str = None):
    zip_path = raw_dir / preferred_name if preferred_name else None
    if zip_path and zip_path.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        return

    for z in raw_dir.glob("*.zip"):
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)


def organize_into_class_folders(cfg, raw_dir: Path, output_dir: Path, val_split=0.15, seed=42):
    labels_path = raw_dir / cfg["labels_csv"]
    if not labels_path.exists():
        # search subdirectories in case Kaggle nested the extracted files
        matches = list(raw_dir.rglob(cfg["labels_csv"]))
        if not matches:
            all_csvs = list(raw_dir.rglob("*.csv"))
            hint = (f"Found these CSVs instead: {[str(p) for p in all_csvs]}\n"
                    f"If you used --kaggle_dataset_slug, the mirror may name its labels "
                    f"file differently — pass --labels_csv_override <name> to point at it."
                    if all_csvs else "No CSV files found under this directory at all.")
            sys.exit(f"Could not find {cfg['labels_csv']} under {raw_dir}.\n{hint}")
        labels_path = matches[0]

    image_dir_matches = list(raw_dir.rglob(cfg["image_dir"]))
    image_dir = image_dir_matches[0] if image_dir_matches else raw_dir / cfg["image_dir"]
    if not image_dir.exists():
        sys.exit(f"Could not find an image folder named '{cfg['image_dir']}' under {raw_dir}.\n"
                  f"Check what actually got downloaded with: find {raw_dir} -maxdepth 3")

    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labeled images from {labels_path}")

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df[cfg["label_col"]]
    )

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            class_id = int(row[cfg["label_col"]])
            class_folder = output_dir / split_name / CLASS_NAMES[class_id]
            class_folder.mkdir(parents=True, exist_ok=True)

            src_name = f"{row[cfg['image_col']]}{cfg['image_ext']}"
            src_path = image_dir / src_name
            dst_path = class_folder / src_name

            if src_path.exists() and not dst_path.exists():
                shutil.copyfile(src_path, dst_path)

        print(f"{split_name}: {len(split_df)} images organized into {output_dir / split_name}")

    print("\nClass distribution (train):")
    print(train_df[cfg["label_col"]].value_counts().sort_index().rename(index=CLASS_NAMES))
    print("\nClass distribution (val):")
    print(val_df[cfg["label_col"]].value_counts().sort_index().rename(index=CLASS_NAMES))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["aptos", "eyepacs"], default="aptos")
    parser.add_argument("--output_dir", default="data", help="Where to write data/train, data/val")
    parser.add_argument("--raw_dir", default=None, help="Where raw Kaggle files are/will be stored")
    parser.add_argument("--skip_download", action="store_true", help="Use files already present in --raw_dir")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument(
        "--kaggle_dataset_slug", default=None,
        help="Download from a Kaggle Dataset (owner/name) instead of the Competition endpoint. "
             "Use this if the competition download 403s — datasets aren't rules-gated.",
    )
    parser.add_argument(
        "--presorted", action="store_true",
        help="Set this if --kaggle_dataset_slug points at a mirror that's already split "
             "into one folder per class (e.g. folders named No_DR/Mild/Moderate/Severe/"
             "Proliferative) rather than a flat image folder + labels CSV.",
    )
    parser.add_argument(
        "--labels_csv_override", default=None,
        help="If a dataset mirror names its labels CSV differently than the original "
             "competition, point at the exact filename here.",
    )
    args = parser.parse_args()

    cfg = dict(DATASET_CONFIG[args.dataset])  # copy so overrides don't mutate the shared config
    if args.labels_csv_override:
        cfg["labels_csv"] = args.labels_csv_override

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(f"data/raw_{args.dataset}")

    if not args.skip_download:
        if args.kaggle_dataset_slug:
            download_dataset_mirror(args.kaggle_dataset_slug, raw_dir)
        else:
            download_dataset(cfg, raw_dir)
    else:
        if not raw_dir.exists():
            sys.exit(f"--skip_download was set but {raw_dir} does not exist.")

    if args.presorted:
        organize_presorted_dataset(raw_dir, output_dir, val_split=args.val_split)
    else:
        organize_into_class_folders(cfg, raw_dir, output_dir, val_split=args.val_split)
    print(f"\nDone. Dataset ready at {output_dir}/train and {output_dir}/val "
          f"in the layout expected by src/preprocessing.py.")


if __name__ == "__main__":
    main()
