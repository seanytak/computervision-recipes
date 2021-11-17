"""
Semantic Segmentation PyTorch Dataset that has masks available as labels
"""
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset


def rgb_mask_to_gt_mask(
    rgb_mask: np.ndarray, rgb_to_label: Dict[Tuple, int]
) -> np.ndarray:
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


class MaskLabelsDataset(Dataset):

    _available_mask_formats = set(["class", "rgb"])

    def __init__(
        self,
        labels_filepath: str,
        mask_format: str = "class",
        class_dict_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        labels_filepath : str
            Path to CSV with columns "image_filepath" and "mask_filepath"
        mask_format : str
            Format the masks are in (class, rgb)
        class_dict_path : str
            Path to a class dictionary.
            If rgb_masks is true, then there should be columns, "r", "g", and "b" that specify
            the colors that correspond to a class
        """
        if mask_format not in self._available_mask_formats:
            raise ValueError(
                f"Parameter mask_format must be one of the following values {self._available_mask_formats}"
            )
        self.labels = pd.read_csv(labels_filepath)
        self.mask_format = mask_format

        if mask_format == "rgb":
            if class_dict_path is None:
                raise ValueError("If masks are RGB, class_dict_path is required")
            class_dict = pd.read_csv(class_dict_path).to_dict("index")
            self.class_id_to_name = {
                class_id: rec["name"] for class_id, rec in class_dict.items()
            }
            self.rgb_to_class = {
                (rec["r"], rec["g"], rec["b"]): int(class_id)
                for class_id, rec in class_dict.items()
            }

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        item = self.labels.iloc[idx]

        image = Image.open(item["image_filepath"]).convert("RGB")
        image = np.array(image).astype("float32")

        mask = Image.open(item["mask_filepath"])
        mask = np.array(mask).astype("uint8")

        if self.mask_format == "rgb":
            mask = rgb_mask_to_gt_mask(mask, self.rgb_to_class)

        return image, mask

    def __len__(self):
        return len(self.labels)
