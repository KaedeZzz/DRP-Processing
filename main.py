from direction import drp_direction_map, drp_mask_angle
from src import ROI, ImagePack
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    roi = ROI(0, 0, 384 * 5, 216 * 5)
    images = ImagePack(roi=roi)
    images.slice_images(3, 3)
    ax = images.plot_drp(images.get_mean_drp(), cmap='jet')
    plt.show()
    mag_map, deg_map = drp_direction_map(images)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=90)
    masked_images = images.mask_images(img_mask, normalize=True)
    for i in range(len(masked_images)):
        if i % 100 == 0:
            masked_images[i].show()
