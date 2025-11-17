import numpy as np
from tqdm import tqdm

def hough_transform(edge_image, rho_res=1, theta_res=1):
    """
    Compute the Hough Transform accumulator for lines.
    
    Parameters:
        edge_image : 2D numpy array (binary edge map)
        rho_res    : resolution of rho in pixels
        theta_res  : resolution of theta in degrees
        
    Returns:
        accumulator : 2D array (rho x theta)
        rhos        : array of rho values
        thetas      : array of theta values (radians)
    """
    # Image dimensions
    h, w = edge_image.shape

    # Theta values (in radians)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))

    # Rho range
    diag_len = int(np.ceil(np.sqrt(h*h + w*w)))
    rhos = np.arange(-diag_len, diag_len, rho_res)

    # Accumulator array; modified from simple Hough such that it does
    #  not only record appearances but also values.
    accumulator = np.zeros((len(rhos), len(thetas)))

    ys, xs = np.nonzero(edge_image)  # Get coordinates of edge points
    print(max(xs), max(ys))
    pixel_list = list(zip(ys, xs))


    # Fill accumulator
    for y, x in tqdm(pixel_list, desc='Performing Hough Transform'): # Iterate over edge points
        for t_idx, theta in enumerate(thetas): # Iterate over angles
            # For each edge pixel, the code finds all possible lines (rho, theta) that could pass through it.
            rho = x * np.cos(theta) + y * np.sin(theta) # Calculates distance from origin to the proposed line.
            rho_idx = int((rho + diag_len) / rho_res) # Find index of this distance
            accumulator[rho_idx, t_idx] += edge_image[y, x] / 255 # Increment normalised value for accumulator at (rho, theta)

    return accumulator, rhos, thetas


def find_hough_peaks(accumulator, num_peaks=5, threshold=1000):
    peaks = []
    acc = accumulator.copy()

    for _ in range(num_peaks):
        idx = np.argmax(acc)
        rho_idx, theta_idx = np.unravel_index(idx, acc.shape)
        
        if acc[rho_idx, theta_idx] < threshold:
            break

        peaks.append((rho_idx, theta_idx, acc[rho_idx, theta_idx]))
        
        # Zero out region around the peak to prevent finding multiple nearby peaks
        acc[max(0, rho_idx-10):rho_idx+10, max(0, theta_idx-10):theta_idx+10] = 0

    return peaks