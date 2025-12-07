from pathlib import Path

import numpy as np

from .config import CacheConfig, DRPConfig, load_drp_config, save_cache_config
from .drp_compute import (
    apply_angle_slice,
    build_drp_stack,
    drp_from_images,
    drp_from_stack,
    mask_images as compute_mask_images,
    mean_drp_from_stack,
)
from .drp_plot import plot_drp
from .image_io import (
    prepare_cache,
    resolve_config_path,
    resolve_image_folder,
    load_images,
)
from .paths import DataPaths


class ImagePack:
    def __init__(
        self,
        folder: str | Path | None = None,
        img_format: str = "jpg",
        angle_slice: tuple[int, int] = (1, 1),
        data_root: str | Path = "data",
        config_path: str | Path | None = None,
        use_cached_stack: bool = True,
    ):
        self.paths = DataPaths.from_root(data_root)
        self.folder = resolve_image_folder(folder, self.paths)
        self.config_path = resolve_config_path(config_path)
        self.base_config: DRPConfig = load_drp_config(self.config_path)

        # Load raw grayscale images from disk
        self.images = load_images(self.folder, img_format)
        self.num_images = len(self.images)
        self.h, self.w = self.images[0].shape

        # Precompute the stack shape for the requested slice to size memmap correctly
        if self.base_config.ph_num % angle_slice[0] != 0 or self.base_config.th_num % angle_slice[1] != 0:
            raise ValueError("Angle slices must evenly divide ph_num and th_num.")

        sliced_ph = self.base_config.ph_num // angle_slice[0]
        sliced_th = self.base_config.th_num // angle_slice[1]
        stack_shape = (self.h, self.w, sliced_ph, sliced_th)
        self.drp_stack, cache_cfg, stack_needs_build = prepare_cache(
            self.paths, angle_slice, stack_shape
        )

        # Use caller's slice preference; rebuild cache if it differs from what's stored.
        cache_slice = (cache_cfg.ph_slice, cache_cfg.th_slice)
        self.angle_slice = angle_slice
        if cache_slice != self.angle_slice:
            # Force recreation with new slice parameters
            self._close_memmap(self.drp_stack)
            self.drp_stack = np.memmap(
                self.paths.cache / "drp.dat",
                dtype="uint8",
                mode="w+",
                shape=stack_shape,
            )
            save_cache_config(
                self.paths.cache / "data_config.yaml",
                CacheConfig(ph_slice=self.angle_slice[0], th_slice=self.angle_slice[1]),
            )
            stack_needs_build = True

        # Apply slicing to images and config
        self.images, self.param = apply_angle_slice(self.images, self.base_config, self.angle_slice)
        self.num_images = len(self.images)

        expected_shape = (self.h, self.w, self.param.ph_num, self.param.th_num)
        if self.drp_stack.shape != expected_shape or not use_cached_stack:
            # Shape mismatch or caller requested rebuild: recreate stack
            self._close_memmap(self.drp_stack)
            self.drp_stack = np.memmap(
                self.paths.cache / "drp.dat",
                dtype="uint8",
                mode="w+",
                shape=expected_shape,
            )
            save_cache_config(
                self.paths.cache / "data_config.yaml",
                CacheConfig(ph_slice=self.angle_slice[0], th_slice=self.angle_slice[1]),
            )
            stack_needs_build = True

        if stack_needs_build:
            build_drp_stack(self.images, self.param, self.drp_stack)

    def __iter__(self):
        return iter((self.images, self.param))

    def slice_images(self, angle_slice: tuple[int, int]):
        self.images, self.param = apply_angle_slice(self.images, self.base_config, angle_slice)
        self.num_images = len(self.images)
        return self.images

    def mask_images(self, mask: np.ndarray, normalize: bool = False):
        self.images = compute_mask_images(self.images, mask, normalize)
        return self.images

    def drp(self, loc, mode: str = "kernel"):
        if mode == "pixel":
            return drp_from_images(self.images, self.param, loc)
        if mode == "kernel":
            return drp_from_stack(self.drp_stack, loc)
        raise ValueError("mode must be 'pixel' or 'kernel'")

    def plot_drp(self, drp_array, cmap: str = "jet", project: str = "stereo", ax=None):
        return plot_drp(drp_array, self.param, cmap=cmap, project=project, ax=ax)

    def get_mean_drp(self, mode: str = "kernel"):
        if mode == "pixel":
            # Vectorized mean across all pixels: reshape stack to [phi, theta, h, w]
            arr = np.stack(self.images, axis=0).reshape(
                self.param.ph_num, self.param.th_num, self.h, self.w
            )
            return arr.mean(axis=(2, 3))
        return mean_drp_from_stack(self.drp_stack)

    def get_drp_stack(self):
        return self.drp_stack

    @staticmethod
    def _close_memmap(memmap_obj):
        """
        Close a NumPy memmap's underlying mmap, ignoring errors.
        """
        try:
            if hasattr(memmap_obj, "_mmap") and memmap_obj._mmap is not None:
                memmap_obj._mmap.close()
        except Exception:
            pass
