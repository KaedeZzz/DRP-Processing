from direction import drp_direction_map, drp_mask_angle
from src import ROI, ImagePack
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    roi = ROI(0, 0, 3840, 2160)
    images = ImagePack(roi=roi, angle_slice=(2, 2))
    # ax = images.plot_drp(images.get_mean_drp(), cmap='jet')
    # plt.show()
    mag_map, deg_map = drp_direction_map(images)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=90, threshold=45)
    plt.imshow(0.5 * (np.sin(np.radians(deg_map)) + 1), cmap='gray')
    plt.show()
    masked_images = images.mask_images(0.5 * (np.sin(np.radians(deg_map)) + 1), normalize=True)
    for i in range(len(masked_images)):
        if i % 30 == 0:
            masked_images[i].show()
