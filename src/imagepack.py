from .imageparam import ImageParam
from .roi import ROI

from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import yaml

cwd = Path.cwd()

class ImagePack(object):
    # def __init__(self, images: list[Image.Image], param: ImageParam):
    #     self.images = images
    #     self.param = param
    #     self.self_check()

    def __init__(self, folder='images', img_format='jpg', roi: ROI | None = None):
        """
        Load sample and background datasets.
        :param exp_param: The experiment parameters, namely elevation and azimuth angle ranges.
        :param folder: The folder where the images are saved.
        :param img_format: The image format to load.
        :param roi: The region of interest, a list of length 4.
        :return: a stack of images, and a 2D array of elevation and azimuth angles of each image.
        """
        with open("config.yaml", 'r') as stream:
            exp_param = yaml.safe_load(stream)

        # Assign parameters locally
        th_min = exp_param['th_min']
        th_max = exp_param['th_max']
        th_num = exp_param['th_num']
        ph_min = exp_param['ph_min']
        ph_max = exp_param['ph_max']
        ph_num = exp_param['ph_num']
        new_param = ImageParam(th_min, th_max, th_num,
                               ph_min, ph_max, ph_num)
        # Set path to load images
        if folder == '':
            path = cwd
        else:
            path = cwd / folder

        # Search for files with corresponding affix
        images = []
        for image_path in tqdm(sorted(path.glob('*.' + img_format)), desc='loading images'):
            try:
                image = Image.open(image_path)
                images.append(image)
            except IOError:
                print(f"Could not open image at path: {image_path}")

        self.num_images = len(images)
        if self.num_images != th_num * ph_num:
            raise ValueError('The number of images loaded does not match angle range')
        # Convert all images into greyscale
        for i in tqdm(range(self.num_images), desc='converting images into greyscale'):
            images[i] = images[i].convert('L')
            if roi is not None:
                imin, jmin, imax, jmax = roi
                images[i] = images[i].crop((imin, jmin, imax, jmax))

        self.images = images
        self.w, self.h = self.images[0].size
        self.param = new_param # Legacy, attemp not to use
        self.__dict__.update(new_param.__dict__) # Copy all attributes from ImageParam

    def __iter__(self):
        return iter((self.images, self.param))

    def self_check(self):
        # assert isinstance(self.param, ImageParam)
        assert isinstance(self.images[0], Image.Image)

    def reset(self):
        self.w, self.h = self.images[0].size
        self.num_images = len(self.images)
        if self.num_images != self.ph_num * self.th_num:
            raise ValueError('Number of images does not match number of angles.')

    def slice_images(self, slice_phi_step: int, slice_theta_step: int) -> list[Image.Image]:
        """
        Reduce image set resolution in DRP angles by selecting one image from each M phi angles and N theta angles.
        Total numbers of angles must be divisible by the slice step of each angle.
        :param slice_phi_step: retain an image from each *phi_step* phi angles.
        :param slice_theta_step: retain an image from each *phi_step* phi angles.
        :return: sliced images and angle profiles.
        """
        if slice_phi_step <= 0 or slice_theta_step <= 0:
            raise ValueError('phi_step and theta_step must be positive.')
        if self.ph_num % slice_phi_step != 0:
            raise ValueError('ph_num must be divisible by phi_step.')
        if self.th_num % slice_theta_step != 0:
            raise ValueError('theta_step must be divisible by theta_step.')
        if self.num_images != self.ph_num * self.th_num:
            raise ValueError('Number of images does not match number of angles.')

        indices = np.array([i for i in range(len(self.images))]).astype(int)
        indices = np.reshape(indices, (self.ph_num, self.th_num))
        indices = indices[0::slice_phi_step, 0::slice_theta_step]
        indices = indices.ravel()
        self.images = [self.images[i] for i in indices]

        self.ph_max -= (slice_phi_step - 1) * self.ph_step
        self.ph_max = int(self.ph_max)
        self.ph_num /= slice_phi_step
        self.ph_num = int(self.ph_num)
        self.ph_step *= slice_phi_step
        self.ph_step = int(self.ph_step)
        self.th_max -= (slice_theta_step - 1) * self.th_step
        self.th_max = int(self.th_max)
        self.th_num /= slice_theta_step
        self.th_num = int(self.th_num)
        self.th_step *= slice_theta_step
        self.th_step = int(self.th_step)

        self.num_images = len(self.images)
        if self.num_images != self.ph_num * self.th_num:
            raise ValueError('Number of images {} does not match number of angles.'.format(self.num_images))
        
        return self.images

    def mask_images(self, mask: np.ndarray, normalize: bool = False) -> list[Image.Image]:
        """
        Apply a mask to all images in the list.
        Input image will have pixel values between 0 and 255. If any pixel value exceeds 255 after masking, it will be clipped.
        :param images: Images to be processed.
        :param mask: A numpy array representing the mask.
        :param roi: Region of interest; when applied, the masked images will be cropped into this region.
        :param normalize: Setting to true would normalize masked images.
        :return: List of processed images.
        """
        res_list = []
        for image in tqdm(self.images, desc='masking images'):
            arr = np.array(image).astype(np.float64)
            if arr.shape != mask.shape:
                raise ValueError("Shape of mask must match region of interest")
            arr *= mask
            if normalize:
                arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            res_img = Image.fromarray(arr)
            res_list.append(res_img)
        return res_list
    
    def drp(self, loc) -> np.ndarray:
        """
        Calculate DRP for a single pixel.
        :param image_pack: ImagePack instance of input images and DRP parameters.
        :param loc: location of the pixel.
        :return: a 2D Numpy array representing the DRP data. [th_num x ph_num]
        """
        x, y = loc
        drp_array = np.zeros((self.ph_num, self.th_num))
        for k in range(self.num_images):
            i = k // self.th_num
            j = k % self.th_num
            drp_array[i, j] = self.images[k].getpixel((x, y))
        return drp_array

    def plot_drp(self, drp_array, cmap='jet', project: str = 'stereo', ax = None) -> plt.Axes:
        """
        Returns a matplotlib axis of a DRP in polar coordinates.
        :param cmap: matplotlib colormap name.
        :param project: projection method used, either 'stereo' or 'direct'.
        :param ax: matplotlib axis to draw on. If None, a new axis will be created.
        """

        if drp_array.shape != (self.ph_num, self.th_num):
            raise ValueError('Input DRP array size does not match image pack parameters.')

        # Normalize pixel value into int if they were float in [0,1]
        if np.issubdtype(drp_array.dtype, np.floating) and drp_array.max() <= 1.0:
            drp_array = (drp_array * 255).astype(np.uint8)

        # Meshgrid of phi and theta
        phi, theta = np.meshgrid(np.linspace(0, 360 + self.ph_step, self.ph_num + 1),
                                np.linspace(self.th_min, self.th_max + self.th_step, self.th_num + 1))

        # Projection mapping, from angles to x-y plane
        if project == "stereo":
            xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
            yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
        elif project == "direct":
            xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
            yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        else:
            raise ValueError("Unknown project type. Use 'stereo' or 'direct'.")

        # create axis if none provided
        if ax is None:
            fig, ax = plt.subplots()
        # plot the color mesh
        h = ax.pcolormesh(xx, yy, drp_array.T, cmap=cmap, shading='auto')
        ax.set_aspect('equal') # This ensures projected mesh is circular, not elliptical
        ax.axis('off')
        plt.colorbar(h, ax=ax)
        return ax
    
    def get_mean_drp(self) -> np.ndarray:
        """
        Calculate the mean DRP over all pixels in the image set.
        :return: a 2D Numpy array representing the mean DRP data. [th_num x ph_num]
        """
        w, h = self.images[0].size
        drp_array = np.zeros((self.ph_num, self.th_num))
        for i in tqdm(range(w * h), desc='Calculating mean DRP'):
            col = i // h
            row = i % h
            drp_array += (self.drp(loc=(col, row)) / (w * h))
        return drp_array