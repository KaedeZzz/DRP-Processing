import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from drp_processing import drp
from utils import ROI

cwd = Path.cwd()

with open("config.yaml", 'r') as stream:
    exp_param = yaml.safe_load(stream)

# Assign parameters locally
th_min = exp_param['th_min']
th_max = exp_param['th_max']
th_num = exp_param['th_num']
ph_min = exp_param['ph_min']
ph_max = exp_param['ph_max']
ph_num = exp_param['ph_num']


def get_drp_direction(drp_mat: np.ndarray, attenuation: float = 1.0) -> list:
    """
    Calculate overall azimuthal direction of DRP array.
    :param drp_mat: the source DRP array
    :param attenuation: attenuation factor to scale the calculated direction vector; 1.0 will keep the vector a sum of difference of each element with mean value of the matrix.
    :return: a vector of length 2 describing the average azimuthal direction of DRP.
    """

    res = [0.0, 0.0]
    mat_mean = np.mean(drp_mat)

    for p in range(ph_num):
        phi_deg = p * 360 / ph_num
        mag = (np.mean(drp_mat[p, :]) - mat_mean) * attenuation
        phi_rad = phi_deg * np.pi / 180
        x, y = mag * np.cos(phi_rad), mag * np.sin(phi_rad)
        res[0] += x
        res[1] += y

    return res


def drp_direction_map(images: list[Image.Image], roi: ROI | None = None, display: bool = True):
    """
    Calculate (and display) the direction map of a DRP over a region of interest.
    :param images: source images to perform DRP calculation.
    :param roi: region of interest.
    :param display: whether to display the direction map on screen.
    :return: direction map in 2D NumPy array [width, height].
    """
    if roi:
        roi.check(images[0].size)
        imin, jmin, imax, jmax = roi
    else:
        imin, jmin, imax, jmax = 0, 0, images[0].size[0], images[0].size[1]

    mag_map = np.zeros([imax - imin, jmax - jmin])
    deg_map = np.zeros([imax - imin, jmax - jmin])
    for k in tqdm(range((imax - imin) * (jmax - jmin)), desc='Generating DRP direction map'):
        i = k // (jmax - jmin)
        j = k % (jmax - jmin)
        col = i + imin
        row = j + jmin
        drp_mat = drp(images, (col, row))
        drp_vector = get_drp_direction(drp_mat, attenuation=1.0)
        x, y = drp_vector
        deg = np.degrees(np.arctan2(y, x))
        mag = np.linalg.norm(drp_vector)
        mag_map[i, j] = mag
        deg_map[i, j] = deg

    # clip extreme values, normalise
    mat_mean = np.mean(mag_map)
    mag_map[mag_map > 2 * mat_mean] = 2 * mat_mean
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    # display the magnitude vector maps
    if display:
        fig, axes = plt.subplots(figsize=(13, 3), ncols=3)
        im1 = axes[0].imshow(images[th_num - 1].crop((imin, jmin, imax, jmax)), cmap="gray")
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_title("ROI image")
        im2 = axes[1].imshow(norm_mag_map, cmap='afmhot')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_title("Normalised DRP magnitudes")
        im3 = axes[2].imshow(deg_map, cmap='hsv')
        axes[2].set_title("DRP angles")
        fig.colorbar(im3, ax=axes[2])
        plt.show()

    return norm_mag_map, deg_map


def drp_mask_angle(images: list[Image.Image], mag_map: np.ndarray, deg_map:np.ndarray, roi: ROI | None, threshold, orientation):
    """
    Obtain a mask of image that highlights pixels with DRP orientations around a designated direction.
    :param images: Source images.
    :param mag_map: Map that indicates magnitude of DRP orientation of sample.
    :param deg_map: Map that indicates angle of DRP orientation of sample.
    :param roi: Region of interest instance.
    :param threshold: Two-sided angle threshold around desired orientation.
    :param orientation: Desired orientation to highlight.
    :return: The image mask and the
    """

    if roi:
        roi.check(images[0].size)
        imin, jmin, imax, jmax = roi
    else:
        imin, jmin, imax, jmax = 0, 0, images[0].size[0], images[0].size[1]

    img_mask = np.zeros([imax - imin, jmax - jmin])
    for k in tqdm(range((imax - imin) * (jmax - jmin)), desc='Masking image with directionality'):
        col = k // (jmax - jmin) + imin
        row = k % (jmax - jmin) + jmin
        i = col - imin
        j = row - jmin
        if orientation - threshold < deg_map[i, j] < orientation + threshold:
            img_mask[i, j] = mag_map[i, j]

    return img_mask, roi


