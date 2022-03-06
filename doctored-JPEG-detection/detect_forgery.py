"""
Implementation of the passive detection algorithm of doctored JPEG image via block artifact grid extraction
"""

import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


# Set Matplotlib display parameters
matplotlib.rcParams['figure.dpi'] = 100

# Parameters
THETA = 10 * (np.pi / 180)
THRESHOLD = 30
BLOCK_SIZE = 8

SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def conv_filter(img, mask):
    """
    Convolution filtering function
    """
    return np.sum(img * mask)


def median_filter(img, mask):
    """
    Median filtering function
    """
    temp = img[mask]

    return np.sort(temp)[len(temp) // 2]


def apply_filter(img, mask, filter_func, axis):
    """
    Applying a filter to an image
    """
    filter_size = np.zeros(img.ndim, dtype=np.int32)
    filter_size[axis] = mask.shape[0] // 2

    filtered_img = np.zeros_like(img, dtype=np.int32)
    for i in range(filter_size[0], img.shape[0] - filter_size[0]):
        for j in range(filter_size[1], img.shape[1] - filter_size[1]):
            filtered_img[i, j] = filter_func(img[i - filter_size[0]: i + filter_size[0] + 1,
                                             j - filter_size[1]: j + filter_size[1] + 1].squeeze(), mask)

    return filtered_img


def preprocess_image(img, theta):
    """
    Image preprocessing function using the Sobel operator
    """
    # Detect image edges
    sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    dy = apply_filter(img, sobel_kernel, filter_func=conv_filter, axis=[0, 1])
    dx = apply_filter(img, sobel_kernel.T, filter_func=conv_filter, axis=[0, 1])

    # Filter image the edges
    d = np.arctan2(np.abs(dy), np.abs(dx))
    r = (theta < d) & (d < np.pi / 2 - theta)

    return r


def extract_bag(img, r, threshold):
    """
    Image block artifact grid (BAG) extraction algorithm implementation
    """
    # Extract vertical and horizontal edges from an image
    second_diff = np.array([-1, 2, -1], dtype=np.int32)
    dy = np.abs(apply_filter(img, second_diff, filter_func=conv_filter, axis=0))
    dx = np.abs(apply_filter(img, second_diff, filter_func=conv_filter, axis=1))
    # Remove any non-vertical/non-horizontal or strong image edges
    dy[dy > threshold] = dy[r] = 0
    dx[dx > threshold] = dx[r] = 0

    # Enlarge weak image's BAG lines
    box_kernel = np.ones(33, dtype=np.int32)
    ey = apply_filter(dy, box_kernel, filter_func=conv_filter, axis=1)
    ex = apply_filter(dx, box_kernel, filter_func=conv_filter, axis=0)
    # Equalize the amplitudes throughout the result image
    mask = box_kernel.astype(bool)
    eh = ey - apply_filter(ey, mask, filter_func=median_filter, axis=0)
    ev = ex - apply_filter(ex, mask, filter_func=median_filter, axis=1)

    # Enhance the BAGs from noise
    mask = np.zeros(33, dtype=bool)
    mask[::8] = True
    gh = apply_filter(eh, mask, filter_func=median_filter, axis=0)
    gv = apply_filter(ev, mask, filter_func=median_filter, axis=1)

    # Get a final BAG image
    g = gv + gh

    return g


def return_anomaly_score(bag_img, block_size):
    """
    BAG image's anomaly score computing function
    """
    anomaly_score = np.zeros_like(bag_img, dtype=np.int32)
    for i in range(bag_img.shape[0] // block_size):
        for j in range(bag_img.shape[1] // block_size):
            img_block = bag_img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            max_col = np.max(img_block[1:-1, 1:-1].sum(axis=0))
            min_col = np.min(img_block[1:-1, [0, -1]].sum(axis=0))
            max_row = np.max(img_block[1:-1, 1:-1].sum(axis=1))
            min_row = np.min(img_block[[0, -1], 1:-1].sum(axis=1))

            # Calculate the anomaly score
            anomaly_score[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = \
                max_col - min_col + max_row - min_row

    return anomaly_score


def display_image(img, title="", save_path=None):
    """
    Image displaying and saving function
    """
    img = np.maximum(img, 0) / img.max()

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)

    if save_path is not None:
        plt.imsave(save_path, img, cmap='gray')


def main():
    """
    Main called function
    """
    # Read and display a source image
    img = cv2.imread(os.path.join(*SOURCE_PATH.split('/')), 0)
    display_image(img, title="Source image")

    # Extract and display an image BAG
    r = preprocess_image(img, theta=THETA)
    bag_img = extract_bag(img, r, threshold=THRESHOLD)
    display_image(bag_img, title="BAG image")

    # Compute, display, and save a BAG image's anomaly score
    anomaly_score = return_anomaly_score(bag_img, block_size=BLOCK_SIZE)
    display_image(anomaly_score, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
