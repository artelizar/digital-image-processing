"""
Implementation of the Sauvola's document image adaptive binarization method
"""

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Parameters
WINDOW_SIZE = 50
K = 0.2
R = 128

SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def calculate_integral_image(img):
    """
    Integral image and integral image square calculating function
    """
    pad_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int64)
    pad_img[1:, 1:] = img

    # Calculate the integral image
    ii = np.cumsum(pad_img, axis=1)
    ii = np.cumsum(ii, axis=0)
    # Calculate the integral image square
    ii_2 = np.cumsum(pad_img ** 2, axis=1)
    ii_2 = np.cumsum(ii_2, axis=0)

    return ii, ii_2


def correct_index(idx, n):
    """
    If an index is outside an image, then it is converted to the closest image index
    """
    # +1 due to adding a row and column of zeros to the image
    return min(max(idx + 1, 0), n)


def Sauvola_binarization(img, window_size, k, r):
    """
    Sauvola's method implementation using integral image calculation
    """
    threshold = np.empty_like(img, dtype=np.float32)
    ii, ii_2 = calculate_integral_image(img)
    # Loop over an image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Calculate correct window coordinates
            a = (correct_index(i - window_size // 2 - 1, img.shape[0]),
                 correct_index(j - window_size // 2 - 1, img.shape[1]))
            b = (correct_index(i - window_size // 2 - 1, img.shape[0]),
                 correct_index(j + window_size // 2, img.shape[1]))
            c = (correct_index(i + window_size // 2, img.shape[0]),
                 correct_index(j + window_size // 2, img.shape[1]))
            d = (correct_index(i + window_size // 2, img.shape[0]),
                 correct_index(j - window_size // 2 - 1, img.shape[1]))

            # Calculate required values
            n = (c[0] - a[0]) * (c[1] - a[1])
            s = ii[c] - ii[b] - ii[d] + ii[a]
            s_2 = ii_2[c] - ii_2[b] - ii_2[d] + ii_2[a]

            # Calculate window's mean and std
            mean = s / n
            std = math.sqrt(s_2 / n - mean ** 2)

            # Calculate a threshold for a current pixel
            t = mean * (1 + k * (std / r - 1))
            threshold[i, j] = t

    return threshold


def display_image(img, title="", save_path=None):
    """
    Image displaying and saving function
    """
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

    # Find local thresholds using Sauvola's method
    thresholds = Sauvola_binarization(img, window_size=WINDOW_SIZE, k=K, r=R)

    # Create, display, and save a binary image
    bin_img = img > thresholds
    display_image(bin_img, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
