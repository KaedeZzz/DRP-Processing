import numpy as np
from matplotlib import pyplot as plt
import cv2

from direction import drp_direction_map, drp_mask_angle
from src import ROI, ImagePack
from src.line_detection import hough_transform, find_hough_peaks


if __name__ == "__main__":

    """Initialise Images and DRP data"""
    roi = ROI(0, 0, 1920, 1080)
    images = ImagePack(roi=roi, angle_slice=(2, 2))

    """Mean Image"""
    # mean_img = np.mean(images.images, axis=0).astype(np.uint8)
    # plt.imshow(mean_img, cmap='gray')
    # plt.title('Mean Image')
    # plt.show()

    """Area mean DRP"""
    ax = images.plot_drp(images.get_mean_drp(), cmap='jet')
    plt.show()

    """Direction Map"""
    mag_map, deg_map = drp_direction_map(images)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=45)
    norm_deg_map = 0.5 * (np.sin(np.radians(deg_map)) + 1)
    # plt.imshow(norm_deg_map.T, cmap='gray')
    # plt.show()
    img = (norm_deg_map * 255).astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()

    """Downsampling and Averaging"""
    img_downsample = img[::2, ::2]
    img_downsample = cv2.blur(img_downsample, (13, 13))
    img_downsample = cv2.blur(img_downsample, (17, 17))
    img_downsample = cv2.blur(img_downsample, (21, 21))
    plt.imshow(img_downsample, cmap='gray')
    plt.show()

    """Downsampling and Gaussian Blurring"""
    # img_downsample = img[::4, ::4]
    # img_downsample = cv2.GaussianBlur(img_downsample, (21, 21), sigmaX=0)
    # plt.imshow(img_downsample.T, cmap='gray')
    # plt.title('Downsampled Image')
    # plt.show()

    """Hough Transform for line detection"""
    # accumulator, rhos, thetas = hough_transform(img_downsample, rho_res=1, theta_res=1)
    # peaks = find_hough_peaks(accumulator, num_peaks=10, threshold=1000)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(accumulator, cmap='hot', aspect=0.02)
    # plt.xlabel('Theta (degrees)')
    # plt.ylabel('Rho (pixels)')
    # plt.title('Hough Transform Accumulator')
    # plt.colorbar()
    # plt.show()

    """Column Stacking Profile"""
    img_stacked = np.mean(img_downsample, axis=0)
    plt.plot(img_stacked)
    plt.title('Stacked Image Intensity Profile')
    plt.xlabel('Pixel Position')
    plt.ylabel('Average Intensity')
    plt.show()
    img_stacked_d = np.convolve(img_stacked, [-0.5, 0, 0.5], mode='same')
    plt.plot(img_stacked_d[2:-2])
    plt.title('First Derivative of Stacked Profile')
    plt.xlabel('Pixel Position')
    plt.ylabel('First Derivative Intensity')
    plt.show()

    """Masking images based on DRP orientation"""
    # masked_images = images.mask_images(0.5 * (np.sin(np.radians(deg_map)) + 1), normalize=True)
    # for i in range(len(masked_images)):
    #     if i % 30 == 0:
    #         masked_images[i].show()
