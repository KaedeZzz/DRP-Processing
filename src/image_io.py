from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .config import (
    CacheConfig,
    DRPConfig,
    load_cache_config,
    load_drp_config,
    save_cache_config,
)
from .paths import DataPaths


def resolve_config_path(config_path: str | Path | None) -> Path:
    """
    Resolve the DRP config path relative to the repository root when not absolute.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = Path(config_path) if config_path is not None else repo_root / "exp_param.yaml"
    return candidate if candidate.is_absolute() else repo_root / candidate


def resolve_image_folder(folder: str | Path | None, paths: DataPaths) -> Path:
    """
    Resolve the folder where images are located, defaulting to the processed path.
    """
    if folder is None:
        return paths.processed
    folder_path = Path(folder)
    return folder_path if folder_path.is_absolute() else paths.root / folder_path


def load_images(folder: Path, img_format: str) -> list[np.ndarray]:
    """
    Load grayscale images from a folder matching the given extension.
    """
    images: list[np.ndarray] = []
    for image_path in tqdm(sorted(folder.glob(f"*.{img_format}")), desc="loading images"):
        image = cv2.imread(str(image_path), flags=cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not open image at path: {image_path}")
        images.append(image)
    if not images:
        raise ValueError(f"No images with extension .{img_format} found in {folder}")
    return images


def open_drp_memmap(path: Path, shape: tuple[int, int, int, int], mode: str) -> np.memmap:
    """
    Open a memmap for the DRP stack, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype="uint8", mode=mode, shape=shape) # type: ignore


def prepare_cache(
    paths: DataPaths,
    angle_slice: tuple[int, int],
    stack_shape: tuple[int, int, int, int],
) -> tuple[np.memmap, CacheConfig, bool]:
    """
    Prepare the DRP cache. Returns (memmap, cache_cfg, is_new_stack).
    is_new_stack is True when the memmap is opened in write mode and needs population.
    """
    data_config_path = paths.cache / "data_config.yaml"
    drp_path = paths.cache / "drp.dat"

    cache_cfg = CacheConfig(ph_slice=angle_slice[0], th_slice=angle_slice[1])
    if data_config_path.exists() and drp_path.exists():
        cache_cfg = load_cache_config(data_config_path)
        memmap = open_drp_memmap(drp_path, stack_shape, mode="r+")
        return memmap, cache_cfg, False

    save_cache_config(data_config_path, cache_cfg)
    memmap = open_drp_memmap(drp_path, stack_shape, mode="w+")
    return memmap, cache_cfg, True
