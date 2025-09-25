from drp_processing import drp_loader, display_drp, area_mean_drp
from direction import drp_direction_map, get_drp_direction, drp_mask_angle
from utils import ROI, mask_images, gamma_transform
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    roi = ROI(0, 0, 100, 50)
    mean_drp_mat = area_mean_drp(images, roi, display=True) # display image selected at highest elevation angle
    display_drp(mean_drp_mat)
    print(get_drp_direction(mean_drp_mat))
    mag_map, deg_map = drp_direction_map(images, roi)
    img_mask = drp_mask_angle(mag_map, deg_map, orientation=0, threshold=30)
    masked_images = mask_images(images, img_mask, roi=roi)
    masked_arrays = [np.array(image) for image in masked_images]
    for i in range(len(masked_arrays)):
        if i % 5 == 0 and i <= 20:
            plt.imshow(masked_arrays[i])
    plt.show()
    transformed_arrays = [gamma_transform(masked_array, return_type='uint8') for masked_array in masked_arrays]
    transformed_images = [Image.fromarray(array) for array in transformed_arrays]
    for i in range(len(transformed_images)):
        if i % 5 == 0 and i <= 20:
            transformed_images[i].show()