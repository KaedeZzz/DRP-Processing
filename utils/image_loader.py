from PIL import Image
from tqdm import tqdm


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
            print(image_path)
            image = Image.open(image_path)
            images.append(image)
        except IOError:
            print(f"Could not open image at path: {image_path}")
    return images