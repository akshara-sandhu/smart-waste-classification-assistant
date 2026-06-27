from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT, RANDOM_SEED


TRASHNET_ZIP_URL = (
    "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading TrashNet dataset...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded to: {destination}")


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    print("Extracting dataset repository...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    # Find the inner dataset-resized.zip file.
    dataset_zip_files = list(
        extract_dir.rglob("dataset-resized.zip")
    )

    if not dataset_zip_files:
        raise FileNotFoundError(
            "Could not find dataset-resized.zip inside the downloaded repository."
        )

    dataset_zip_path = dataset_zip_files[0]
    dataset_dir = dataset_zip_path.parent / "dataset-resized"

    print("Extracting the image dataset...")

    with zipfile.ZipFile(dataset_zip_path, "r") as dataset_archive:
        dataset_archive.extractall(dataset_zip_path.parent)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder was not created: {dataset_dir}"
        )

    return dataset_dir


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    random.seed(RANDOM_SEED)

    class_dirs = sorted(
        path for path in source_dir.iterdir() if path.is_dir()
    )

    if not class_dirs:
        raise ValueError("No class folders were found in the dataset.")

    for split_name in ("train", "val", "test"):
        for class_dir in class_dirs:
            (output_dir / split_name / class_dir.name).mkdir(
                parents=True,
                exist_ok=True,
            )

    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in split_map.items():
            destination_dir = output_dir / split_name / class_dir.name

            for image_path in split_images:
                shutil.copy2(
                    image_path,
                    destination_dir / image_path.name,
                )

        print(
            f"{class_dir.name}: "
            f"{len(split_map['train'])} train, "
            f"{len(split_map['val'])} val, "
            f"{len(split_map['test'])} test"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the downloaded ZIP and extracted files.",
    )
    args = parser.parse_args()

    download_dir = PROJECT_ROOT / "_dataset_download"
    zip_path = download_dir / "trashnet.zip"
    extract_dir = download_dir / "extracted"

    download_file(TRASHNET_ZIP_URL, zip_path)
    dataset_dir = extract_zip(zip_path, extract_dir)
    split_dataset(dataset_dir, DATA_DIR)

    if not args.keep_download:
        shutil.rmtree(download_dir, ignore_errors=True)

    print()
    print("Dataset preparation complete.")
    print(f"Prepared data folder: {DATA_DIR}")
    print("Next command:")
    print("python -m src.train --epochs 10 --batch-size 32")


if __name__ == "__main__":
    main()
