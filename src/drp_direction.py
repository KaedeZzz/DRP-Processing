from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .imagepack import ImagePack


def get_drp_direction(drp_mat: np.ndarray, ph_num: int, attenuation: float = 1.0) -> np.ndarray:
    """
    Calculate overall azimuthal direction of a DRP array.
    Returns a 2D vector representing the weighted average direction.
    """
    mat_mean = np.mean(drp_mat)
    phi_angles = np.linspace(0, 2 * np.pi, ph_num, endpoint=False)
    mag = (drp_mat.mean(axis=1) - mat_mean) * attenuation
    x = np.sum(mag * np.cos(phi_angles))
    y = np.sum(mag * np.sin(phi_angles))
    return np.array([x, y])


def drp_direction_map(imp: "ImagePack", display: bool = True):
    """
    Calculate (and optionally display) the direction map of a DRP over all pixels.
    Returns magnitude and angle maps.
    """
    h, w = imp.h, imp.w
    phi_vec = np.mean(imp.drp_stack, axis=3)  # [h, w, ph_num]
    mean_mat = np.mean(phi_vec, axis=2, keepdims=True)
    phi_angles = np.linspace(0, 2 * np.pi, imp.param.ph_num, endpoint=False)[:, None]
    phi_cos = np.cos(phi_angles)
    phi_sin = np.sin(phi_angles)

    # Project onto unit circle basis for each pixel
    X = (phi_vec - mean_mat) @ phi_cos
    Y = (phi_vec - mean_mat) @ phi_sin
    X = X.reshape(h, w)
    Y = Y.reshape(h, w)
    mag_map = np.sqrt(X**2 + Y**2)
    deg_map = np.degrees(np.arctan2(Y, X))

    # Clip extreme values and normalise magnitude
    mat_mean = np.mean(mag_map)
    mag_map = np.clip(mag_map, None, 2 * mat_mean)
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    if display:
        fig, axes = plt.subplots(figsize=(13, 4), ncols=3)
        im1 = axes[0].imshow(imp.images[0], cmap="gray")
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_title("ROI image")
        im2 = axes[1].imshow(norm_mag_map, cmap="afmhot")
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_title("Normalised DRP magnitudes")
        im3 = axes[2].imshow(deg_map, cmap="hsv")
        axes[2].set_title("DRP angles")
        fig.colorbar(im3, ax=axes[2])
        plt.show()

    return norm_mag_map, deg_map


def drp_mask_angle(mag_map: np.ndarray, deg_map: np.ndarray, orientation: float, threshold: float) -> np.ndarray:
    """
    Create a mask highlighting pixels with DRP orientations around a designated direction.
    Mask values are magnitude-weighted in [0,1].
    """
    if mag_map.shape != deg_map.shape:
        raise ValueError("Magnitude and orientation dimensions do not match.")

    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    # Smallest angular difference to target orientation in degrees
    angle_diff = np.abs(((deg_map - orientation + 180) % 360) - 180)
    mask = angle_diff <= threshold
    return norm_mag_map * mask
