import numpy as np

def gamma_transform(img: np.ndarray, gamma: float = 2, epsilon: int = 5, linear_coeff: float = 2, return_type: str = 'float64')-> np.ndarray:
    """Perform gamma transform on an image. Return type can be 'float64' or 'uint8'."""
    if img.dtype == np.uint8:
        img += epsilon
        img = img.astype(np.float64)
        img /= 255
    elif img.dtype == np.float64:
        img += epsilon / 255
    else:
        raise TypeError('mask dtype must be either np.uint8 or np.float64')

    img = np.power(img, gamma)
    img *= linear_coeff

    if return_type == 'float64':
        img = np.clip(img, 0, 1).astype(np.float64)
    elif return_type == 'uint8':
        img *= 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise TypeError('return_type must be either "float64" or "uint8"')
    return img