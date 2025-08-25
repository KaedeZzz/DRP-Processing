"""
This file contains functions for DRM dataset processing.
There are three parts included in the file:
raw image loader -- load the images in time sequence to form basic image stack
background subtraction -- converting image stack into DRP cell
plotting -- plot DRPs.
"""

import yaml

from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
from scipy.signal import wiener

from utils import load_images

cwd = Path.cwd()

with open("config.yaml", 'r') as stream:
    exp_param = yaml.safe_load(stream)

def drp_loader(folder='images', img_format='jpg', roi: list | np.array = None):
    """
    Load sample and background datasets.
    :param exp_param: The experiment parameters, namely elevation and azimuth angle ranges.
    :param folder: The folder where the images are saved.
    :param img_format: The image format to load.
    :param roi: The region of interest.
    :return: a stack of images, and a 2D array of elevation and azimuth angles of each image.
    """

    # Assign parameters locally
    th_min = exp_param['th_min']
    th_max = exp_param['th_max']
    th_num = exp_param['th_num']
    ph_min = exp_param['ph_min']
    ph_max = exp_param['ph_max']
    ph_num = exp_param['ph_num']

    if len(roi) != 4:
        raise ValueError('Region of interest must contain 4 coordinates')

    # Set path to load images
    if folder == '':
        path = cwd
    else:
        path = cwd / folder

    images = load_images(path, img_format)
    num_images = len(images)

    if num_images != th_num * ph_num:
        raise ValueError('The number of images loaded does not match angle range')

    # Convert all images into greyscale
    for i in tqdm(range(num_images), desc='converting images into greyscale'):
        images[i] = images[i].convert('L')
        if roi is not None:
            images[i] = images[i].crop(*roi)

    # Construct a profile of (phi, theta) angles for each image
    ph_th_profile = np.zeros((num_images, 2))
    indexing = np.arange(1, num_images + 1)
    phi_step = (ph_max - ph_min) / (ph_num - 1)
    th_step = (th_max - th_min) / (th_num - 1)
    for i in range(num_images):
        ph_th_profile[i, 0] = ((indexing[i] - 1) // th_num) * phi_step
        ph_th_profile[i, 1] = ((indexing[i] - 1) % th_num) * th_step + th_min

    return images, ph_th_profile


def bg_subtraction(samples: list[Image.Image], backgrounds: list[Image.Image], coeff: float = 1.0) -> list[Image.Image]:
    """
    Apply background subtraction to images.
    :param samples: list of all images.
    :param backgrounds: list of all backgrounds.
    :param coeff: coefficient of subtraction.
    :return: a list of normalized images after background subtraction.
    """
    num_samples = len(samples)
    if num_samples != len(backgrounds):
        raise ValueError('The number of samples does not match number of backgrounds')

    # Apply Wiener filter to the backgrounds, the window size parameter is from original MATLAB code
    for i in tqdm(range(num_samples), desc='applying wiener filter'):
        img = backgrounds[i]
        img_array = np.array(img)
        filtered_array = wiener(img_array, mysize=(7, 7))
        filtered_bg = Image.fromarray(np.uint8(filtered_array))
        backgrounds[i] = filtered_bg

    # Normalize images by division by backgrounds
    norm_images = []
    for i in tqdm(range(num_samples), desc='normalizing images'):
        sample = np.array(samples[i]).astype(np.float32)
        back = np.array(backgrounds[i]).astype(np.float32)
        back[back == 0] = 1 # Prevent divide-by-zero
        norm = (sample / back) / coeff * 255
        norm_images.append(Image.fromarray(np.uint8(norm)))

    return norm_images

if __name__ == '__main__':
    images, profile = drp_loader()
    print(profile)