from .imageparam import ImageParam
from .roi import ROI
from .paths import DataPaths

import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import yaml


class ImagePack(object):
    # def __init__(self, images: list[Image.Image], param: ImageParam):
    #     self.images = images
    #     self.param = param
    #     self.self_check()

    def __init__(
        self,
        folder: str | Path | None = None,
        img_format: str = 'jpg',
        angle_slice: tuple[int, int] = (1, 1),
        data_root: str | Path = "data",
        config_path: str | Path | None = None,
    ):
        """
        Load sample datasets. Images will be converted into greyscale automatically.
        :param folder: folder path containing images to be loaded.
        :param img_format: image file format/extension.
        :param roi: region of interest to crop images into. If None, full images will be used.
        :param read_mode: if True, then DRP data will be directly read from file.
        :param slice: tuple of (phi_slice, theta_slice) to reduce image set resolution in DRP angles. Only required when read_mode is False.
        """
        paths = DataPaths.from_root(data_root)
        self.paths = paths

        # Resolve path to load images
        if folder is None:
            path = paths.processed
        else:
            folder_path = Path(folder)
            path = folder_path if folder_path.is_absolute() else paths.root / folder_path

        # Resolve experiment config path (defaults to repo root exp_param.yaml)
        repo_root = Path(__file__).resolve().parent.parent
        config_path = Path(config_path) if config_path is not None else repo_root / "exp_param.yaml"
        if not config_path.is_absolute():
            config_path = repo_root / config_path

        with open(config_path, 'r') as stream:
            exp_param = yaml.safe_load(stream)

        # Assign parameters locally
        th_min = exp_param['th_min']
        th_max = exp_param['th_max']
        th_num = exp_param['th_num']
        ph_min = exp_param['ph_min']
        ph_max = exp_param['ph_max']
        ph_num = exp_param['ph_num']
        ph_slice = exp_param.get('phi_slice', 1)
        th_slice = exp_param.get('theta_slice', 1)
        new_param = ImageParam(th_min, th_max, th_num,
                               ph_min, ph_max, ph_num)

        # Search for files with corresponding affix
        images = []
        for image_path in tqdm(sorted(path.glob('*.' + img_format)), desc='loading images'):
            try:
                image = cv2.imread(str(image_path), flags=cv2.IMREAD_GRAYSCALE)
                images.append(image)
            except IOError:
                print(f"Could not open image at path: {image_path}")
        self.num_images = len(images)
        
        # Convert all images into greyscale
        # for i in tqdm(range(self.num_images), desc='converting images into greyscale'):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        #     if roi is not None:
        #         imin, jmin, imax, jmax = roi
        #         images[i] = images[i][jmin:jmax, imin:imax]

        # Assign attributes
        self.images = images
        print(self.images[0].shape)
        self.h, self.w = self.images[0].shape
        self.param = new_param # Legacy, attemp not to use
        self.__dict__.update(new_param.__dict__) # Copy all attributes from ImageParam
        self.drp_path = paths.cache / 'drp.dat'
        data_config_path = paths.cache / 'data_config.yaml'

        if data_config_path.exists() and self.drp_path.exists():
            print('Reading DRP data from file...')
            with open(data_config_path, 'r') as file:
                saved_config = yaml.safe_load(file)
                ph_slice = saved_config.get('ph_slice', 1)
                th_slice = saved_config.get('th_slice', 1)
                angle_slice = (ph_slice, th_slice)
                self.slice_images(angle_slice)
                shape = (self.h, self.w, self.ph_num, self.th_num)   
                drp_stack = np.memmap(self.drp_path, dtype='uint8', mode='r+', shape=shape)
                self.drp_stack = drp_stack
            
        else:
            # Write drp_config.yaml and generate drp.dat if not in read mode
            config = {
                "th_min": self.th_min,
                "th_max": self.th_max,
                "th_num": self.th_num,
                "ph_min": self.ph_min,
                "ph_max": self.ph_max,
                "ph_num": self.ph_num,
                "ph_slice": angle_slice[0],
                "th_slice": angle_slice[1],
            }
            paths.cache.mkdir(parents=True, exist_ok=True)
            with open(data_config_path, 'w') as file:
                yaml.dump(config, file)
            self.slice_images(angle_slice)
            shape = (self.w, self.h, self.ph_num, self.th_num)
            self.get_drp_stack()

        return
    

    def __iter__(self):
        return iter((self.images, self.param))
    
    def slice_indices(self, angle_slice: tuple[int, int]) -> np.ndarray:
        (ph_slice, th_slice) = angle_slice
        if ph_slice <= 0 or th_slice <= 0:
            raise ValueError('phi_step and theta_step must be positive.')
        if self.ph_num % ph_slice != 0:
            raise ValueError('ph_num must be divisible by phi_step.')
        if self.th_num % th_slice != 0:
            raise ValueError('theta_num {} must be divisible by theta_step {}.', self.th_num, th_slice)
        indices = np.array([i for i in range(len(self.images))]).astype(int)
        indices = np.reshape(indices, (self.ph_num, self.th_num))
        indices = indices[0::ph_slice, 0::th_slice]
        indices = indices.ravel()
        return indices
    

    def slice_images(self, angle_slice: tuple[int, int]) -> list[np.ndarray]:
        """
        Reduce image set resolution in DRP angles by selecting one image from each M phi angles and N theta angles.
        Total numbers of angles must be divisible by the slice step of each angle.
        :param slice_phi_step: retain an image from each *phi_step* phi angles.
        :param slice_theta_step: retain an image from each *phi_step* phi angles.
        :return: sliced images and angle profiles.
        """
        if self.num_images != self.ph_num * self.th_num:
            raise ValueError('Number of images does not match number of angles.')
        
        ph_slice, th_slice = angle_slice
        indices = self.slice_indices(angle_slice)
        self.images = [self.images[i] for i in indices]

        self.ph_max -= (ph_slice - 1) * self.ph_step
        self.ph_max = int(self.ph_max)
        self.ph_num /= ph_slice
        self.ph_num = int(self.ph_num)
        self.ph_step *= ph_slice
        self.ph_step = int(self.ph_step)
        self.th_max -= (th_slice - 1) * self.th_step
        self.th_max = int(self.th_max)
        self.th_num /= th_slice
        self.th_num = int(self.th_num)
        self.th_step *= th_slice
        self.th_step = int(self.th_step)

        self.num_images = len(self.images)
        if self.num_images != self.ph_num * self.th_num:
            raise ValueError('Number of images {} does not match number of angles.'.format(self.num_images))
        
        return self.images


    def mask_images(self, mask: np.ndarray, normalize: bool = False) -> list[np.ndarray]:
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
            if arr.shape != mask.T.shape:
                print(arr.shape, mask.T.shape)
                raise ValueError("Shape of mask must match region of interest")
            arr *= mask.T
            if normalize:
                arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            res_list.append(arr)
        return res_list
    

    def drp(self, loc, mode='kernel') -> np.ndarray:
        """
        Calculate DRP for a single pixel.
        :param image_pack: ImagePack instance of input images and DRP parameters.
        :param loc: location of the pixel.
        :return: a 2D Numpy array representing the DRP data. [th_num x ph_num]
        """
        y, x = loc
        drp_array = np.zeros((self.ph_num, self.th_num))
        if mode == 'pixel':
            for k in range(self.num_images):
                i = k // self.th_num
                j = k % self.th_num
                drp_array[i, j] = self.images[k][y, x]
        elif mode == 'kernel':
            drp_array = self.drp_stack[y, x, :, :]
        return drp_array


    def plot_drp(self, drp_array, cmap='jet', project: str = 'stereo', ax = None):
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
    

    def get_mean_drp(self, mode='kernel') -> np.ndarray:
        """
        Calculate the mean DRP over all pixels in the image set.
        :return: a 2D Numpy array representing the mean DRP data. [th_num x ph_num]
        """
        w, h = self.images[0].shape
        drp_array = np.zeros((self.ph_num, self.th_num))
        if mode == 'pixel':
            for i in tqdm(range(w * h), desc='Calculating mean DRP'):
                col = i // h
                row = i % h
                drp_array += self.drp(loc=(col, row), mode='pixel') / (w * h)
        elif mode == 'kernel':
            drp_array = np.mean(self.drp_stack, axis=(0,1))
        return drp_array


    def get_drp_stack(self) -> None:
        """
        Generate a 4D NumPy array representing the DRP data for all pixels.
        """
        shape = (self.h, self.w, self.ph_num, self.th_num)
        drp_stack = np.memmap(self.drp_path, dtype='uint8', mode='w+', shape=shape)

        for i in tqdm(range(self.h * self.w), desc='Generating DRP stack'):
            col = i // self.h
            row = i % self.h
            drp_stack[row, col, :, :] = self.drp(loc=(row, col), mode='pixel')
            if i % 50 == 0: # Beware of memory limit! Do not load too much per flush.
                drp_stack.flush()
        self.drp_stack = drp_stack
        return
    
