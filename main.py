from drp_processing import drp_loader, display_drp, area_mean_drp
from direction import drp_direction_map, get_drp_direction, drp_mask_angle
from utils import ROI, mask_images, gamma_transform
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    roi = ROI(0, 0, 3840, 2160)
    mean_drp_mat = area_mean_drp(images, roi, display=True) # display image selected at highest elevation angle
    display_drp(mean_drp_mat)
    mag_map, deg_map = drp_direction_map(images, roi)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=30)
    masked_images = mask_images(images, img_mask, roi=roi, normalize=True)
    for i in range(len(images)):
        if i % 100 == 0:
            masked_images[i].show()
