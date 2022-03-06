"""
Implementation of the Otsu's image global binarization method
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


L = 256  # 8-bit grayscale color

# Parameters
SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def calculate_hist(img):
    """
    Image histogram calculating function
    """
    values, counts = np.unique(img, return_counts=True)
    hist = np.zeros(L, dtype=np.int32)
    hist[values] = counts

    return hist


def Otsu_thresholding(img):
    """
    Otsu's method implementation
    """
    # Calculate image histogram
    hist = calculate_hist(img)
    # Find borders of the image on the histogram
    lid, rid = 0, L - 1
    for i in range(L):
        if hist[i]:
            lid = i
            break
    for i in range(L):
        if hist[L - 1 - i]:
            rid = L - 1 - i
            break
    # Normalize the image histogram as a probability distribution
    normalized_hist = np.asarray(hist / hist.sum(), dtype=np.float32)

    # Calculate the mathematical expectation of an image pixel
    mt = 0
    for i in range(lid, rid + 1):
        mt += i * normalized_hist[i]

    # Calculate the maximum between-class variance (object and background classes) and a binarization threshold
    max_vb, threshold = 0, -1
    w0, m0 = 0, 0
    for t in range(lid, rid):
        w0 += normalized_hist[t]
        m0 += normalized_hist[t] * t

        vb = w0 * (1 - w0) * ((mt - m0) / (1 - w0) - m0 / w0) ** 2
        if vb > max_vb:
            max_vb = vb
            threshold = t

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

    # Find a global threshold using Otsu's method
    threshold = Otsu_thresholding(img)
    print("Global threshold =", threshold)

    # Create, display, and save a binary image
    bin_img = np.array(img > threshold, dtype=np.uint8) * (L - 1)
    display_image(bin_img, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
