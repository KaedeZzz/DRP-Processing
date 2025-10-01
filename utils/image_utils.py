from PIL import Image
from tqdm import tqdm
import numpy as np

class ROI:
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


def load_images(path, affix: str) -> list[Image.Image]:
    """
    Open all images in a folder and return as a list.
    :param path: Path to the folder.
    :param affix: File format to open, dot excluded.
    :return: A list of image objects of class Image.Image.
    """
    images = []
    for image_path in tqdm(sorted(path.glob('*.' + affix)), desc='loading images'):
        try:
            image = Image.open(image_path)
            images.append(image)
        except IOError:
            print(f"Could not open image at path: {image_path}")
    return images


def mask_images(images: list[Image.Image], mask: np.ndarray, roi: ROI | None, normalize: bool = False) -> list[Image.Image]:
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
    for image in tqdm(images, desc='masking images'):
        arr = np.array(image).astype(np.float64)
        if roi:
            imin, jmin, imax, jmax = roi.x, roi.y, roi.x + roi.w, roi.y + roi.h
            arr = arr[imin:imax, jmin:jmax]
        if arr.shape != mask.shape:
            raise ValueError("Shape of mask must match region of interest")
        arr *= mask
        if normalize:
            arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        res_img = Image.fromarray(arr)
        res_list.append(res_img)
    return res_list