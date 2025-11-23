from tqdm import tqdm
import pathlib
import numpy as np
import cv2
import sys, os


def rgb2grey(img, mode='luminosity'):
    """Convert RGB image to greyscale using luminosity method."""
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        if mode == 'average':
            grey_img = np.mean(img, axis=2)
        elif mode == 'luminosity':
            grey_img = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]
            grey_img = np.reshape(grey_img, (img.shape[0], img.shape[1]))
        return grey_img.astype(np.uint8)
    else:
        raise ValueError("Input image must be either a 2D greyscale or a 3D RGB image.")
    

if __name__ == "__main__":
    # Search for files with corresponding affix
    roi = None # region of interest of the images to be processed
    read_path = pathlib.Path.cwd() / 'sample' # image path to read
    write_path = pathlib.Path.cwd() / 'processed' # image path to write
    print(f"Reading images from folder: {read_path}")
    print(f"Saving processed images to folder: {write_path}")
    img_format = 'jpg'

    images = []
    sample_paths = sorted(read_path.glob('*.' + img_format))
    for image_path in tqdm(sample_paths, desc='loading images'):
        try:
            image = cv2.imread(str(image_path))
            images.append(image)
        except IOError:
            print(f"Could not open image at path: {image_path}")
    num_images = len(images)
    
    # Convert all images into greyscale
    for i in tqdm(range(num_images), desc='converting images into greyscale'):
        images[i] = rgb2grey(images[i], mode='luminosity')
        if roi is not None:
            imin, jmin, imax, jmax = roi
            images[i] = images[i][jmin:jmax, imin:imax]

    write_path.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(num_images), desc='saving greyscale images'):
        cv2.imwrite(str(write_path / sample_paths[i].name), images[i])