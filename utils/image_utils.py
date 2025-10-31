from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import yaml

cwd = Path.cwd()


class ROI(object):
    def __init__(self, x, y, w, h):
        """
        Region of interest bounding box.
        :param x: Horizontal position of upperleft pixel
        :param y: Vertical position of upperleft pixel
        :param w: Width of ROI
        :param h: Height of ROI
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.self_check()

    def __iter__(self):
        return iter((self.x, self.y, self.x + self.w, self.y + self.h))

    def self_check(self):
        if self.x < 0 or self.y < 0:
            raise ValueError("For RoI, x and y must be positive")
        if self.w < 0:
            raise ValueError("For RoI, width must be positive")
        if self.h < 0:
            raise ValueError("For RoI, height must be positive")

    def check(self, dims):
        self.self_check()
        if len(dims) != 2:
            raise ValueError("image must have 2 dimensions")
        elif self.x + self.w > dims[0] or self.y + self.h > dims[1]:
            raise ValueError("RoI exceeds image dimensions")


class ImageParam(object):
    def __init__(self, th_min: int, th_max: int, th_num: int,
                 ph_min: int, ph_max: int, ph_num: int):
        self.ph_min = ph_min
        self.ph_max = ph_max
        self.ph_num = ph_num
        self.ph_step = (ph_max - ph_min) / (ph_num - 1)
        self.th_min = th_min
        self.th_max = th_max
        self.th_num = th_num
        self.th_step = (th_max - th_min) / (th_num - 1)

    def __str__(self):
        return "Current image set DRP parameters:\n"\
                + "phi_min: " + str(self.ph_min) + "\n"\
                + "phi_max: " + str(self.ph_max) + "\n"\
                + "phi_num: " + str(self.ph_num) + "\n"\
                + "phi_step: " + str(self.ph_step) + "\n"\
                + "th_min: " + str(self.th_min) + "\n"\
                + "th_max: " + str(self.th_max) + "\n"\
                + "th_num: " + str(self.th_num) + "\n"\
                + "th_step: " + str(self.th_step)


class Images(object):
    # def __init__(self, images: list[Image.Image], param: ImageParam):
    #     self.images = images
    #     self.param = param
    #     self.self_check()

    def __iter__(self):
        return iter((self.images, self.param))

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

        num_images = len(images)
        if num_images != th_num * ph_num:
            raise ValueError('The number of images loaded does not match angle range')
        # Convert all images into greyscale
        for i in tqdm(range(num_images), desc='converting images into greyscale'):
            images[i] = images[i].convert('L')
            if roi is not None:
                imin, jmin, imax, jmax = roi
                images[i] = images[i].crop((imin, jmin, imax, jmax))
        self.images = images
        self.param = new_param

    def self_check(self):
        assert isinstance(self.param, ImageParam)
        assert isinstance(self.images[0], Image.Image)

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
        if self.param.ph_num % slice_phi_step != 0:
            raise ValueError('ph_num must be divisible by phi_step.')
        if self.param.th_num % slice_theta_step != 0:
            raise ValueError('theta_step must be divisible by theta_step.')
        if len(self.images) != self.param.ph_num * self.param.th_num:
            raise ValueError('Number of images does not match number of angles.')

        indices = np.array([i for i in range(len(self.images))]).astype(int)
        indices = np.reshape(indices, (self.param.ph_num, self.param.th_num))
        indices = indices[0::slice_phi_step, 0::slice_theta_step]
        indices = indices.ravel()
        self.images = [self.images[i] for i in indices]

        self.param.ph_max -= (slice_phi_step - 1) * self.param.ph_step
        self.param.ph_max = int(self.param.ph_max)
        self.param.ph_num /= slice_phi_step
        self.param.ph_num = int(self.param.ph_num)
        self.param.ph_step *= slice_phi_step
        self.param.ph_step = int(self.param.ph_step)
        self.param.th_max -= (slice_theta_step - 1) * self.param.th_step
        self.param.th_max = int(self.param.th_max)
        self.param.th_num /= slice_theta_step
        self.param.th_num = int(self.param.th_num)
        self.param.th_step *= slice_theta_step
        self.param.th_step = int(self.param.th_step)
        # print(self.param)
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