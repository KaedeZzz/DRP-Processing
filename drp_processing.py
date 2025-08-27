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
from matplotlib import pyplot as plt

from utils import load_images

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


def build_angle_profile(num_images):
    # Construct a profile of (phi, theta) angles for each image
    if num_images != th_num * ph_num:
        raise ValueError('The number of images loaded does not match angle range')
    ph_th_profile = np.zeros((num_images, 2))
    indexing = np.arange(1, num_images + 1)
    phi_step = (ph_max - ph_min) / (ph_num - 1)
    th_step = (th_max - th_min) / (th_num - 1)
    for i in range(num_images):
        ph_th_profile[i, 0] = ((indexing[i] - 1) // th_num) * phi_step
        ph_th_profile[i, 1] = ((indexing[i] - 1) % th_num) * th_step + th_min
    return ph_th_profile


def drp_loader(folder='images', img_format='jpg', roi: list | np.ndarray = None):
    """
    Load sample and background datasets.
    :param exp_param: The experiment parameters, namely elevation and azimuth angle ranges.
    :param folder: The folder where the images are saved.
    :param img_format: The image format to load.
    :param roi: The region of interest.
    :return: a stack of images, and a 2D array of elevation and azimuth angles of each image.
    """

    if roi and len(roi) != 4:
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

    return images, build_angle_profile(num_images)


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


# def igrey2drp(samples: list[Image.Image]):
#     """
#     Builds list of images into a 4D DRP dataset of dimensions [w x h x theta x phi].
#     :param samples: List of image samples.
#     :return: A numpy array of dimension 4.
#     """
#     width, height = samples[0].size
#     dataset = np.zeros((width, height, th_num, ph_num))
#     angle_profile = build_angle_profile(len(samples))
#     for i in tqdm(range(len(samples)), desc='building 4D DRP dataset'):
#         phi, theta = angle_profile[i]
#         phi_step = (ph_max - ph_min) / (ph_num - 1)
#         th_step = (th_max - th_min) / (th_num - 1)
#         phi_ind = int((phi - ph_min) / phi_step)
#         theta_ind = int((theta - th_min) / th_step)
#         dataset[:, :, phi_ind, theta_ind] = np.array(samples[i]).T
#
#     return dataset


def display_drp(drp_array: np.ndarray, cmap='jet', project: str = 'stereo', ax = None, scalebar: bool = True):
    """
    Returns a matplotlib axis of a DRP in polar coordinates.
    :param drp_array: 2D dataset in [theta, phi].
    :param cmap: matplotlib colormap name.
    :param project: projection method used, either 'stereo' or 'direct'.
    :param ax: matplotlib axis to draw on.
    :param scalebar: boolean, whether to display a scalebar on the plot.
    :return: a matplotlib axis of the DRP in polar coordinates.
    """
    # Normalize pixel value into int if they were float in [0,1]
    if np.issubdtype(drp_array.dtype, np.floating) and drp_array.max() <= 1.0:
        drp_array = (drp_array * 255).astype(np.uint8)

    # Meshgrid of phi (x-axis), theta (y-axis)
    ph_step = 360 / ph_num
    th_step = (th_max - th_min) / th_num
    phi, theta = np.meshgrid(np.linspace(0, 360 + ph_step, ph_num + 1),
                             np.linspace(th_min, th_max + th_step, th_num + 1))

    # Projection mapping
    if project == "stereo":
        xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
        yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
    elif project == "direct":
        xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    else:
        raise ValueError("Unknown project type. Use 'stereo' or 'direct'.")

    if ax is None:
        fig, ax = plt.subplots()

    print(drp_array.shape)

    h = ax.pcolormesh(xx, yy, drp_array, cmap=cmap, shading='auto')
    ax.set_aspect('equal')
    ax.axis('off')

    if scalebar:
        plt.colorbar(h, ax=ax)

    return ax

def drp_measure(img_sample, dataset: np.ndarray=None, images: list[Image.Image]=None):
    fig, ax = plt.subplots()
    ax.imshow(img_sample, cmap="gray")
    ax.set_title("Click on image for DRP points, press ENTER to stop")
    points = plt.ginput(n=-1, timeout=30)  # unlimited clicks, press Enter to stop
    plt.close(fig)

    if len(points) == 0:
        print("No points selected.")
        return []

    num_points = len(points)
    x = [int(p[0]) for p in points]
    y = [int(p[1]) for p in points]

    fig, axs = plt.subplots(num_points + 1)
    # First subplot shows original image with markers
    axs[0].imshow(img_sample, cmap="gray")
    axs[0].scatter(x, y, c='r', marker='x', s=100)
    for i in range(num_points):
        axs[0].text(x[i] + 5, y[i] + 5, str(i + 1), fontsize=12, color='yellow')
    axs[0].set_title("Selected Points")

    drp_measurement = []
    for i in tqdm(range(num_points), desc='calculating DRP measurements'):
        row, col = x[i], y[i]
        drp_list = [images[k].getpixel((row, col)) for k in range(len(images))]
        drp = np.reshape(drp_list, (ph_num, th_num))
        drp = drp.T
        # drp = dataset[row][col]  # assuming drp_original is a 2D list
        drp_measurement.append(drp)

        # Show DRP using custom display function
        axs[i + 1] = display_drp(drp, ax=axs[i + 1])
        axs[i + 1].set_title(f"DRP of point {i + 1}")

    plt.tight_layout()
    plt.show()

    return drp_measurement

if __name__ == '__main__':
    images, profile = drp_loader()
    print(profile)