from glob import glob
from os.path import basename, join
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple
from src.datasets.coco import BoundingBox

from torch.utils.data import Dataset


def rgb_mask_to_gt_mask(rgb_mask: np.ndarray, rgb_to_label: Dict[Tuple, int]):
    """
    Parameters
    ----------
    mask : np.ndarray
        RGB Mask where each color indicates a specific class
        Shape `(height, width, 3)`
    rgb_to_label : dict of tuples to ints
        Dictionary where an RGB tuple maps to its class label

    Returns
    -------
    mask : np.ndarray
        Mask converted from RGB labels to class labels
        Shape `(height, width)`
    """

    def rgb_to_int(arr):
        """Create a hashcode for an RGB value by the formula
        R * 256**2 + G * 256 + B

        Convert (N,...M,3)-array of dtype uint8 to a (N,...,M)-array of dtype int32
        """
        return arr[..., 0] * (256 ** 2) + arr[..., 1] * 256 + arr[..., 2]

    int_colors = rgb_to_int(rgb_mask)
    int_keys = rgb_to_int(np.array(list(rgb_to_label.keys()), dtype="uint8"))
    int_array = np.r_[int_colors.ravel(), int_keys]
    _, index = np.unique(int_array, return_inverse=True)
    color_labels = index[: int_colors.size]
    key_labels = index[-len(rgb_to_label) :]

    colormap = np.empty_like(int_keys, dtype="uint32")
    colormap[key_labels] = list(rgb_to_label.values())
    mask = colormap[color_labels].reshape(rgb_mask.shape[:2])
    return mask


class DroneDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, class_dict_path: str):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_index = [
            basename(filename).split(".")[0]
            for filename in glob(join(images_dir, "*.jpg"))
        ]

        class_dict = pd.read_csv(class_dict_path).to_dict("index")
        self.class_id_to_name = {
            class_id: rec["name"] for class_id, rec in class_dict.items()
        }
        self.rgb_to_class = {
            (rec["r"], rec["g"], rec["b"]): int(class_id)
            for class_id, rec in class_dict.items()
        }

    def __getitem__(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        filename = self.images_index[image_id]
        image_filepath = join(self.images_dir, f"{filename}.jpg")
        image = Image.open(image_filepath).convert("RGB")
        image = np.array(image).astype("float32")

        mask_filepath = join(self.masks_dir, f"{filename}.png")
        mask = Image.open(mask_filepath)  # .convert("RGB")
        mask = np.array(mask).astype("uint8")

        # mask = rgb_mask_to_gt_mask(mask, self.rgb_to_class)

        return image, mask

    def __len__(self):
        return len(self.images_index)
