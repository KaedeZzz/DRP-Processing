import numpy as np
import scipy
from matplotlib import pyplot as plt
import cv2

from paperdrm import ImagePack, Settings
from paperdrm.drp_direction import drp_direction_map, drp_mask_angle

import cv2
from paperdrm.line_detection import (
        hough_transform,
        find_hough_peaks,
        dominant_orientation_from_accumulator,
        rotate_image_to_orientation,
        overlay_hough_lines,
    )


if __name__ == "__main__":
    # Centralised settings loaded from YAML with a small override for angle slicing
    settings = Settings.from_yaml("exp_param.yaml").with_overrides(angle_slice=(2, 2))

    # Load images + DRP (2x2 angular slice)
    images = ImagePack(settings=settings)

    # Mean DRP plot
    ax = images.plot_drp(images.get_mean_drp(), cmap="jet")
    plt.show()

    # Direction map + mask
    mag_map, deg_map = drp_direction_map(images)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=45)

    # Normalize orientation to 0–255 and show
    img = (0.5 * (-np.sin(np.radians(deg_map)) + 1) * 255).astype(np.uint8)
    plt.imshow(img, cmap="gray")
    plt.show()

    # Background subtraction on the deg_map-derived image to suppress low-frequency bias.
    img_float = img.astype(np.float32)
    bg_sigma = max(3.0, min(img.shape) / 50.0)
    background = cv2.GaussianBlur(img_float, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    diff = img_float - background
    limit = np.percentile(np.abs(diff), 99.0)
    limit = max(limit, 1.0)
    diff = np.clip(diff / limit, 0, 1.0)
    img = (diff * 255).astype(np.uint8)
    plt.imshow(img, cmap="gray")
    plt.title("Background-subtracted deg_map image")
    plt.show()

    # Downsample first, then a single Gaussian blur instead of multiple box blurs
    img = img[::2, ::2]
    img = cv2.GaussianBlur(img, (11, 11), sigmaX=4, sigmaY=4)
    plt.imshow(img, cmap="gray")
    plt.show()

    # Column intensity profile
    img_stacked = img.mean(axis=0)
    plt.plot(img_stacked)
    plt.title("Stacked Image Intensity Profile")
    plt.xlabel("Pixel Position")
    plt.ylabel("Average Intensity")
    plt.show()

    # Peak detection (top 13 by height) and overlay
    peaks, _ = scipy.signal.find_peaks(img_stacked)
    peaks_by_height = peaks[np.argsort(img_stacked[peaks])[::-1]]
    keep = 25
    plt.imshow(img, cmap="gray")
    for peak in peaks_by_height[:keep]:
        plt.plot([peak, peak], [0, img.shape[0]], color="red", linewidth=1)
    plt.show()
