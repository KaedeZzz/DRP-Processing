import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
        for t in range(th_num):
            theta_deg = t * 360 / th_num
            mag = (drp_mat[t, p] - mat_mean) * attenuation
            theta_rad = theta_deg * np.pi / 180
            x, y = mag * np.cos(theta_rad), mag * np.sin(theta_rad)
            res[0] += x
            res[1] += y

    return res


def drp_direction_map(images: list[Image.Image], roi: list | np.ndarray = None, display: bool = True) -> np.ndarray:
    """
    Calculate (and display) the direction map of a DRP over a region of interest.
    :param images: source images to perform DRP calculation.
    :param roi: region of interest.
    :param display: whether to display the direction map on screen.
    :return: direction map in 2D NumPy array [width, height].
    """
    if roi and len(roi) != 4:
        raise ValueError("ROI must be of length 4")
    elif roi:
        imin, imax, jmin, jmax = roi
    else:
        imin, imax, jmin, jmax = 0, images[0].size[0], 0, images[0].size[1]
    num_points = len(images)

    drp_array_4 = np.zeros([imax - imin, jmax - jmin, th_num, ph_num]) # 4 Dimensions!
    for i in tqdm(range((imax - imin) * (jmax - jmin)), desc='calculating pixel-wise DRP'):
        row = i // (jmax - jmin)
        col = i % (jmax - jmin)
        drp_list = [images[k].getpixel((row, col)) for k in range(len(images))]
        drp = np.reshape(drp_list, (ph_num, th_num))
        drp = drp.T  # in consistency with custom display function
        drp_array_4[row, col] = drp

    mag_map = np.zeros([imax - imin, jmax - jmin])
    deg_map = np.zeros([imax - imin, jmax - jmin])
    for i in tqdm(range((imax - imin) * (jmax - jmin)), desc='calculating DRP direction vectors'):
        row = i // (jmax - jmin)
        col = i % (jmax - jmin)
        drp_vector = get_drp_direction(drp_array_4[row, col], attenuation=1.0)
        x, y = drp_vector
        deg = np.degrees(np.arctan2(y, x))
        mag = np.linalg.norm(drp_vector)
        mag_map[row, col] = mag
        deg_map[row, col] = deg

    # clip extreme values, normalise
    mat_mean = np.mean(mag_map)
    mag_map[mag_map > 2 * mat_mean] = 2 * mat_mean
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    # display the magnitude vector maps
    if display:
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(images[0].crop((imin, jmin, imax, jmax)), cmap="gray")
        axes[0].set_title("ROI image")
        axes[1].imshow(norm_mag_map, cmap='jet')
        axes[1].set_title("Normalised DRP magnitudes")
        axes[2].imshow(deg_map, cmap='hsv')
        axes[2].set_title("DRP angles")
        plt.show()

    return norm_mag_map, deg_map