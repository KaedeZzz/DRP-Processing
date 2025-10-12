import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from drp_processing import drp
from utils import ImagePack, ImageParam

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


def get_drp_direction(drp_mat: np.ndarray, params: ImageParam, attenuation: float = 1.0) -> list:
    """
    Calculate overall azimuthal direction of DRP array.
    :param drp_mat: the source DRP array
    :param attenuation: attenuation factor to scale the calculated direction vector; 1.0 will keep the vector a sum of difference of each element with mean value of the matrix.
    :return: a vector of length 2 describing the average azimuthal direction of DRP.
    """

    res = [0.0, 0.0]
    mat_mean = np.mean(drp_mat)

    for p in range(params.ph_num):
        phi_deg = p * 360 / params.ph_num
        mag = (np.mean(drp_mat[p, :]) - mat_mean) * attenuation
        phi_rad = phi_deg * np.pi / 180
        x, y = mag * np.cos(phi_rad), mag * np.sin(phi_rad)
        res[0] += x
        res[1] += y

    return res


def drp_direction_map(image_pack: ImagePack, display: bool = True):
    """
    Calculate (and display) the direction map of a DRP over a region of interest.
    :param image_pack: ImagePack instance of images and DRP parameters.
    :param display: whether to display the direction map on screen.
    :return: direction map in 2D NumPy array [width, height].
    """
    images, params = image_pack
    w, h = images[0].size
    mag_map = np.zeros((w, h))
    deg_map = np.zeros((w, h))
    for k in tqdm(range(w * h), desc='Generating DRP direction map'):
        i = k // h
        j = k % h
        drp_mat = drp(image_pack, (i, j))
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
        fig, axes = plt.subplots(figsize=(13, 4), ncols=3)
        im1 = axes[0].imshow(images[th_num - 1], cmap="gray")
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_title("ROI image")
        im2 = axes[1].imshow(norm_mag_map.T, cmap='afmhot')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_title("Normalised DRP magnitudes")
        im3 = axes[2].imshow(deg_map.T, cmap='hsv')
        axes[2].set_title("DRP angles")
        fig.colorbar(im3, ax=axes[2])
        plt.show()

    return norm_mag_map, deg_map


def drp_mask_angle(mag_map: np.ndarray, deg_map:np.ndarray, orientation, threshold) -> np.ndarray:
    """
    Obtain a mask of image that highlights pixels with DRP orientations around a designated direction.
    :param mag_map: Map that indicates magnitude of DRP orientation of sample.
    :param deg_map: Map that indicates angle of DRP orientation of sample.
    :param threshold: Two-sided angle threshold around desired orientation.
    :param orientation: Desired orientation to highlight.
    :return: The image mask and the
    """

    if mag_map.shape != deg_map.shape:
        raise ValueError('Magnitude and orientation dimensions do not match.')

    # magnitude needs to be normalised before use
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    w, h = mag_map.shape[:2]
    img_mask = np.ones((w, h)) / 255
    for k in range(w * h):
        i = k // h
        j = k % h
        if orientation - threshold < deg_map[i, j] < orientation + threshold\
            or orientation - threshold < deg_map[i, j] + 360 < orientation + threshold\
            or orientation - threshold < deg_map[i, j] - 360 < orientation + threshold:
            img_mask[i, j] = norm_mag_map[i, j]
    return img_mask


