"""
This is a preprocessing script specific for organizing the drone dataset images
It creates the labels csv file that is required by the core training script

Example Trial:
python preprocess.py \
    --images-dir workshop_data/images \
    --masks-dir workshop_data/masks \
    --train-labels-dir workshop_data/train_labels \
    --val-labels-dir workshop_data/val_labels
"""
import argparse
import csv
import hashlib
from pathlib import Path
from os.path import basename, join
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--train-labels-dir")
    parser.add_argument("--val-labels-dir")
    args = parser.parse_args()

    images_dir = str(args.images_dir)
    masks_dir = str(args.masks_dir)
    train_labels_dir = str(args.train_labels_dir)
    val_labels_dir = str(args.val_labels_dir)

    train_data = []
    val_data = []

    for img_filepath in glob(join(images_dir, "*.jpg")):
        filename, _ = basename(img_filepath).split(".")
        mask_filepath = join(masks_dir, f"{filename}.png")

        # On average this should uniformally split the data
        # between train and val by 80/20
        rem = int(hashlib.md5(img_filepath.encode()).hexdigest(), 16) % 5
        if rem == 0:
            val_data.append((img_filepath, mask_filepath))
        else:
            train_data.append((img_filepath, mask_filepath))

    def write_labels(labels, labels_dir):
        labels_csv = Path(join(labels_dir, "labels.csv"))
        labels_csv.parent.mkdir(exist_ok=True)
        with open(labels_csv, "w") as f:
            write = csv.writer(f)
            write.writerow(["image_filepath", "mask_filepath"])
            write.writerows(labels)

    write_labels(train_data, train_labels_dir)
    write_labels(val_data, val_labels_dir)
