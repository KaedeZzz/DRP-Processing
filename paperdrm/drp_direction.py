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


def drp_direction_map(imp: "ImagePack", display: bool = True, verbose: bool = False):
    """
    Calculate (and optionally display) the direction map of a DRP over all pixels.
    Returns magnitude and angle maps.
    """
    if verbose:
        print("[DRP] computing DRP direction map")
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

    if verbose:
        print(f"[DRP] direction map computed; mag_map range=({norm_mag_map.min():.3f}, {norm_mag_map.max():.3f})")
    return norm_mag_map, deg_map


def drp_mask_angle(
    mag_map: np.ndarray,
    deg_map: np.ndarray,
    orientation: float,
    threshold: float,
    *,
    verbose: bool = False,
) -> np.ndarray:
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
    if verbose:
        keep_pct = 100 * np.mean(mask)
        print(f"[DRP] mask around {orientation}±{threshold} keeps {keep_pct:.1f}% of pixels")
    return norm_mag_map * mask


def spherical_descriptor(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel first-order (vector) and second-order (covariance) spherical moments.

    Returns:
        m1_map: [h, w, 3] first-order moment (dominant 3D direction, not normalised).
        cov_map: [h, w, 3, 3] second-order spread around m1.
        weight_sum: [h, w] sum of absolute weights used for normalisation.
    """
    # Stack layout: [h, w, phi, theta]
    if verbose:
        print("spherical_descriptor: preparing stack")
    stack = imp.drp_stack.astype(np.float64)
    if subtract_mean:
        stack = stack - stack.mean(axis=(2, 3), keepdims=True)

    # Angle grids in radians (inclusive endpoints to match ph/th counts)
    phi = np.deg2rad(np.linspace(imp.param.ph_min, imp.param.ph_max, imp.param.ph_num, endpoint=True))
    theta = np.deg2rad(np.linspace(imp.param.th_min, imp.param.th_max, imp.param.th_num, endpoint=True))
    phi_grid = phi[:, None]  # [phi, 1]
    theta_grid = theta[None, :]  # [1, theta]

    # Unit direction vectors for each (phi, theta) sample
    sin_theta = np.sin(theta_grid)
    sin_phi = np.sin(phi_grid)
    cos_theta = np.cos(theta_grid)
    cos_phi = np.cos(phi_grid)
    dir_grid = np.stack(
        [
            cos_theta * cos_phi,  # x
            cos_theta * sin_phi,  # y
            np.broadcast_to(sin_theta, (imp.param.ph_num, imp.param.th_num)),  # z
        ],
        axis=-1,
    )  # [phi, theta, 3]

    # Weights: DRP values (minus mean) optionally scaled by sin(theta) for spherical area element
    if verbose:
        print("spherical_descriptor: computing weights")
    weight = stack
    if include_sin_theta:
        weight = weight * np.sin(theta_grid)

    # Normalise weights per pixel to avoid bias from brightness
    if verbose:
        print("spherical_descriptor: normalising weights")
    weight_sum = np.sum(np.abs(weight), axis=(2, 3), keepdims=True)
    norm_weight = weight / (weight_sum + eps)

    # First-order moment (3D vector)
    if verbose:
        print("spherical_descriptor: computing first-order moment")
    m1_map = np.sum(norm_weight[..., None] * dir_grid[None, None, ...], axis=(2, 3))

    # Centered directions for covariance
    if verbose:
        print("spherical_descriptor: computing covariance")
    centered = dir_grid[None, None, ...] - m1_map[..., None, None, :]
    cov_map = np.einsum(
        "hwpt,hwpti,hwptj->hwij",
        norm_weight,
        centered,
        centered,
    )

    if verbose:
        print("spherical_descriptor: done")
    return m1_map, cov_map, weight_sum[..., 0, 0]


def spherical_descriptor_maps(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper to compute direction, projection, and strength maps
    from per-pixel spherical moments.

    Returns:
        dir_map: [h, w, 3] unit vectors (normalised first-order moment).
        proj_map: [h, w, 2] azimuth/elevation in degrees (azimuth from +x, elevation from xy-plane).
        strength_map: [h, w] magnitude of the first-order moment (0→isotropic).
        m1_map: [h, w, 3] raw first-order moment before normalisation.
        cov_map: [h, w, 3, 3] covariance (second-order spread).
    """
    m1_map, cov_map, _ = spherical_descriptor(
        imp,
        subtract_mean=subtract_mean,
        include_sin_theta=include_sin_theta,
        eps=eps,
        verbose=verbose,
    )

    # Direction (unit vector) and strength (norm)
    strength_map = np.linalg.norm(m1_map, axis=2, keepdims=True)
    dir_map = m1_map / (strength_map + eps)

    # Project to azimuth/elevation in degrees
    az_map = np.degrees(np.arctan2(dir_map[..., 1], dir_map[..., 0]))
    el_map = np.degrees(np.arcsin(np.clip(dir_map[..., 2], -1.0, 1.0)))
    proj_map = np.stack([az_map, el_map], axis=-1)

    # strength_map back to [h, w]
    strength_map = strength_map[..., 0]
    return dir_map, proj_map, strength_map, m1_map, cov_map


def _sorted_eigh_3x3(cov_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetrise and eigen-decompose a per-pixel 3x3 covariance.
    Eigenvalues are ascending; eigenvectors are column-wise.
    """
    cov_sym = 0.5 * (cov_map + np.swapaxes(cov_map, -1, -2))
    evals, evecs = np.linalg.eigh(cov_sym)
    return evals, evecs


def anisotropy_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Per-pixel anisotropy score from covariance eigenvalues.
    High when one principal axis dominates (line-like), low when isotropic.
    Uses (lambda_max - lambda_mid) / (lambda_sum + eps). Shape: [h, w].
    """
    evals, _ = _sorted_eigh_3x3(cov_map)
    lam0, lam1, lam2 = evals[..., 0], evals[..., 1], evals[..., 2]
    lam_sum = lam0 + lam1 + lam2
    score = (lam2 - lam1) / (lam_sum + eps)
    return score


def plane_orientation_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """
    Dominant in-plane orientation and tilt from the leading eigenvector of covariance.

    Returns:
        az_map: [h, w] azimuth of dominant axis projected to xy-plane (degrees).
        el_map: [h, w] elevation of dominant axis (degrees).
    """
    _, evecs = _sorted_eigh_3x3(cov_map)
    v = evecs[..., :, -1]  # dominant eigenvector (largest eigenvalue)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    xy_norm = np.sqrt(vx**2 + vy**2)
    az_map = np.degrees(np.arctan2(vy, vx))
    az_map = np.where(xy_norm > eps, az_map, np.nan)  # undefined if projection is degenerate

    el_map = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
    return az_map, el_map
