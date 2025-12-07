from .config import DRPConfig, CacheConfig, load_drp_config, load_cache_config, save_cache_config
from .roi import ROI
from .paths import DataPaths

import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


class ImagePack(object):

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
        paths = DataPaths.from_root(data_root)  # Initialise data paths
        self.paths = paths

        # Resolve path to load images
        if folder is None:
            path = paths.processed  # By default, use final-processed image folder
        else:
            folder_path = Path(folder)
            path = folder_path if folder_path.is_absolute() else paths.root / folder_path

        # Resolve experiment config path (defaults to repo root exp_param.yaml)
        repo_root = Path(__file__).resolve().parent.parent  # The repo root directory is two levels up from this file
        config_path = Path(config_path) if config_path is not None else repo_root / "exp_param.yaml"
        if not config_path.is_absolute():
            config_path = repo_root / config_path

        self.param = load_drp_config(config_path)  # Build DRPConfig from config file
        ph_slice = self.param.phi_slice
        th_slice = self.param.theta_slice

        # Search for files with corresponding affix
        images = []
        for image_path in tqdm(sorted(path.glob('*.' + img_format)), desc='loading images'):
            try:
                image = cv2.imread(str(image_path), flags=cv2.IMREAD_GRAYSCALE)
                images.append(image)
            except IOError:
                print(f"Could not open image at path: {image_path}")
        self.num_images = len(images)

        # Assign attributes
        self.images = images
        print(self.images[0].shape)
        self.h, self.w = self.images[0].shape
        self.drp_path = paths.cache / 'drp.dat'
        data_config_path = paths.cache / 'data_config.yaml'

        if data_config_path.exists() and self.drp_path.exists():
            # Use existing drp.dat and data_config.yaml if in read mode
            print('Reading DRP data from file...')
            saved_config = load_cache_config(data_config_path)
            ph_slice = saved_config.ph_slice
            th_slice = saved_config.th_slice
            angle_slice = (ph_slice, th_slice)
            self.slice_images(angle_slice)
            shape = (self.h, self.w, self.param.ph_num, self.param.th_num)
            drp_stack = np.memmap(self.drp_path, dtype='uint8', mode='r+', shape=shape)
            self.drp_stack = drp_stack
            
        else:
            # Write drp_config.yaml and generate drp.dat if not in read mode
            paths.cache.mkdir(parents=True, exist_ok=True)
            save_cache_config(
                data_config_path,
                CacheConfig(ph_slice=angle_slice[0], th_slice=angle_slice[1]),
            )
            self.slice_images(angle_slice)
            shape = (self.w, self.h, self.param.ph_num, self.param.th_num)
            self.get_drp_stack()

        return
    

    def __iter__(self):
        return iter((self.images, self.param))
    
    def slice_indices(self, angle_slice: tuple[int, int]) -> np.ndarray:
        """
        Get indices of images to be retained after slicing.
        :param slice_phi_step: retain an image from each *phi_step* phi angles.
        :param slice_theta_step: retain an image from each *phi_step* phi angles.
        :return: indices of images to be retained.
        """

        (ph_slice, th_slice) = angle_slice
        if ph_slice <= 0 or th_slice <= 0:
            raise ValueError('phi_step and theta_step must be positive.')
        if self.param.ph_num % ph_slice != 0:
            raise ValueError('ph_num must be divisible by phi_step.')
        if self.param.th_num % th_slice != 0:
            raise ValueError('theta_num {} must be divisible by theta_step {}.', self.param.th_num, th_slice)
        indices = np.array([i for i in range(len(self.images))]).astype(int)
        indices = np.reshape(indices, (self.param.ph_num, self.param.th_num))
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
        if self.num_images != self.param.ph_num * self.param.th_num:
            raise ValueError('Number of images does not match number of angles.')
        
        ph_slice, th_slice = angle_slice
        indices = self.slice_indices(angle_slice)
        self.images = [self.images[i] for i in indices]

        # Update DRPConfig accordingly
        self.param.ph_max = int(self.param.ph_max - (ph_slice - 1) * self.param.ph_step)
        self.param.ph_num = int(self.param.ph_num / ph_slice)
        self.param.th_max = int(self.param.th_max - (th_slice - 1) * self.param.th_step)
        self.param.th_num = int(self.param.th_num / th_slice)

        # TODO: save new config back to DRPConfig file

        self.num_images = len(self.images)
        # Validate new image count
        if self.num_images != self.param.ph_num * self.param.th_num:
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
        Calculate DRP for a single pixel. In pixel mode, the DRP is calculated by collecting pixel values from all images at the specified location.
        In kernel mode, the DRP is retrieved directly from the precomputed DRP stack.
        :param image_pack: ImagePack instance of input images and DRP parameters.
        :param loc: location of the pixel.
        :return: a 2D Numpy array representing the DRP data. [th_num x ph_num]
        """
        y, x = loc
        drp_array = np.zeros((self.param.ph_num, self.param.th_num))
        if mode == 'pixel':
            for k in range(self.num_images):
                i = k // self.param.th_num
                j = k % self.param.th_num
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

        if drp_array.shape != (self.param.ph_num, self.param.th_num):
            raise ValueError('Input DRP array size does not match image pack parameters.')

        # Normalize pixel value into int if they were float in [0,1]
        if np.issubdtype(drp_array.dtype, np.floating) and drp_array.max() <= 1.0:
            drp_array = (drp_array * 255).astype(np.uint8)

        # Meshgrid of phi and theta
        phi, theta = np.meshgrid(np.linspace(0, 360 + self.param.ph_step, self.param.ph_num + 1),
                                np.linspace(self.param.th_min, self.param.th_max + self.param.th_step, self.param.th_num + 1))

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
        drp_array = np.zeros((self.param.ph_num, self.param.th_num))
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
        shape = (self.h, self.w, self.param.ph_num, self.param.th_num)
        drp_stack = np.memmap(self.drp_path, dtype='uint8', mode='w+', shape=shape)

        for i in tqdm(range(self.h * self.w), desc='Generating DRP stack'):
            col = i // self.h
            row = i % self.h
            drp_stack[row, col, :, :] = self.drp(loc=(row, col), mode='pixel')
            if i % 50 == 0: # Beware of memory limit! Do not load too much per flush.
                drp_stack.flush()
        self.drp_stack = drp_stack
        return
    
