"""
Preprocessing utilities (former standalone scripts) integrated into the package.

- convert_rgb_to_greyscale: convert raw RGB captures to greyscale into data/processed.
- generate_blurred_backgrounds: build heavy-blur backgrounds into data/background.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .paths import DataPaths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _clear_image_folder(path: Path) -> None:
    """Remove existing image files so reruns start clean."""
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            item.unlink(missing_ok=True)


def _rgb2grey_array(img: np.ndarray, mode: str = "luminosity") -> np.ndarray:
    """Convert RGB image to greyscale using luminosity (default) or average weighting."""
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        if mode == "average":
            grey_img = img.mean(axis=2)
        elif mode == "luminosity":
            grey_img = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]
            grey_img = grey_img.reshape((img.shape[0], img.shape[1]))
        else:
            raise ValueError("Mode not recognised. Use 'average' or 'luminosity'.")
        return grey_img.astype(np.uint8)
    raise ValueError("Input image must be either a 2D greyscale or a 3D RGB image.")


def convert_rgb_to_greyscale(
    data_root: str | Path = "data",
    img_format: str = "jpg",
    mode: str = "luminosity",
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Convert RGB images in data/raw to greyscale in data/processed.
    """
    paths = DataPaths.from_root(data_root)
    read_path = paths.raw
    write_path = paths.processed
    sample_paths = sorted(read_path.glob(f"*.{img_format}"))
    if not sample_paths:
        raise ValueError(f"No images with extension .{img_format} found in {read_path}")

    images: list[np.ndarray] = []
    for image_path in tqdm(sample_paths, desc="loading images"):
        img = cv2.imread(str(image_path))
        if img is None:
            raise IOError(f"Could not open image at path: {image_path}")
        images.append(img)

    for i in tqdm(range(len(images)), desc="converting images into greyscale"):
        images[i] = _rgb2grey_array(images[i], mode=mode)
        if roi is not None:
            imin, jmin, imax, jmax = roi
            images[i] = images[i][jmin:jmax, imin:imax]

    write_path.mkdir(parents=True, exist_ok=True)
    _clear_image_folder(write_path)
    for i in tqdm(range(len(images)), desc="saving greyscale images"):
        cv2.imwrite(str(write_path / sample_paths[i].name), images[i])


def generate_blurred_backgrounds(
    data_root: str | Path = "data",
    img_format: str = "jpg",
    debug_first: bool = False,
) -> None:
    """
    Build heavily blurred backgrounds from processed images into data/background.
    """
    paths = DataPaths.from_root(data_root)
    images: list[np.ndarray] = []
    img_paths = sorted(paths.processed.glob(f"*.{img_format}"))
    if not img_paths:
        raise ValueError(f"No images with extension .{img_format} found in {paths.processed}")

    for idx, img_path in enumerate(tqdm(img_paths, desc="blurring backgrounds")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read {img_path}")
        img = cv2.GaussianBlur(img, (0, 0), 5)
        small = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (0, 0), 20)
        lowpass = cv2.resize(blurred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        images.append(lowpass)

        # Optional debug visualisation for the first image only.
        if debug_first and idx == 0:
            img_arr = img.astype(np.float32)
            low_arr = lowpass.astype(np.float32)
            diff = img_arr - low_arr
            plt.hist(diff.ravel(), bins=100)
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].set_title("Original Image")
            ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
            ax[1].set_title("Blurred Background")
            ax[1].imshow(lowpass, cmap="gray", vmin=0, vmax=255)
            plt.show()
            plt.imshow(diff, cmap="gray")
            plt.title("Difference Image")
            plt.colorbar()
            plt.show()

    background_dir = paths.root / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    _clear_image_folder(background_dir)
    for i in tqdm(range(len(images)), desc="saving blurred images"):
        cv2.imwrite(str(background_dir / img_paths[i].name), images[i])
