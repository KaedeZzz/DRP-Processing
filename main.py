from drp_processing import display_drp, area_mean_drp
from direction import drp_direction_map, drp_mask_angle
from utils import ROI, Images
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    roi = ROI(0, 0, 384, 216)
    images = Images(roi=roi)
    images.slice_images(3, 3)
    images, params = images
    mean_drp_mat = area_mean_drp(images, display=True) # display image selected at highest elevation angle
    plt.imshow(mean_drp_mat)
    display_drp(mean_drp_mat, params)
    plt.show()
    mag_map, deg_map = drp_direction_map(images)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=30)
    masked_images = Images.mask_images(img_mask, normalize=True)
    for i in range(len(images)):
        if i % 100 == 0:
            masked_images[i].show()
