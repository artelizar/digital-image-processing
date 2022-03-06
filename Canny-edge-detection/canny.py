"""
Implementation of the Canny Edge Detection algorithm
"""

import os
import numpy as np
import cv2
from ccl import BIASES, connected_component_labeling
import matplotlib.pyplot as plt


# Parameters
LOW_THRESHOLD = 10
HIGH_THRESHOLD = 90
WEAK_GRAYSCALE = 85
STRONG_GRAYSCALE = 255

SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def non_maximum_suppression(g, d):
    """
    Leaving edges with the maximum value of a gradient intensity in edge directions
    """
    res = np.zeros_like(g)
    for i in range(1, g.shape[0] - 1):
        for j in range(1, g.shape[1] - 1):
            idx = ((int(d[i, j] // (np.pi / 8)) + 1) // 2) % len(BIASES)
            if g[i, j] > max(g[i + BIASES[idx][0], j + BIASES[idx][1]], g[i - BIASES[idx][0], j - BIASES[idx][1]]):
                res[i, j] = g[i, j]

    return res


def double_thresholding(img, low_threshold, high_threshold, weak_grayscale, strong_grayscale):
    """
    Double thresholding aims at identifying 3 types of pixels: strong, weak, and irrelevant
    """
    res = np.zeros_like(img, dtype=np.uint8)
    res[(img >= low_threshold) & (img <= high_threshold)] = weak_grayscale
    res[img > high_threshold] = strong_grayscale

    return res


def hysteresis_thresholding(img, labeled_img, strong_grayscale):
    """
    The hysteresis is the conversion of weak pixels into strong pixels
    if at least one of pixels in a connected component is strong
    """
    res = np.zeros_like(img)
    for label in np.unique(labeled_img):
        indices = (labeled_img == label)

        mx = img[indices].max()
        if mx == strong_grayscale:
            res[indices] = mx

    return res


def Canny_edge_detection(img, low_threshold, high_threshold, weak_grayscale, strong_grayscale):
    """
    Canny Edge Detection algorithm implementation using the Two-Pass Connected Component Labeling (CCL) algorithm
    """
    # Reduce image noise using the Gaussian blur
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate edge intensities of image objects using the Sobel operator
    dx = cv2.Sobel(blur_img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(blur_img, cv2.CV_32F, 0, 1)
    g = np.sqrt(dx ** 2 + dy ** 2)
    # Calculate edge directions
    d = np.arctan2(dy, dx)
    d[d < 0.0] += np.pi
    # Perform the Non-Maximum Suppression technique to thin out the edges
    suppressed_img = non_maximum_suppression(g, d)

    # Filter the edges into strong and weak
    filtered_img = double_thresholding(suppressed_img, low_threshold, high_threshold, weak_grayscale, strong_grayscale)
    # Hysteresis thresholding using the CCL algorithm
    labeled_img = connected_component_labeling(filtered_img)
    res = hysteresis_thresholding(filtered_img, labeled_img, strong_grayscale)

    return res


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

    # Detect, display, and save edges of image objects using the Canny Edge Detection algorithm
    edges = Canny_edge_detection(img, low_threshold=LOW_THRESHOLD, high_threshold=HIGH_THRESHOLD,
                                 weak_grayscale=WEAK_GRAYSCALE, strong_grayscale=STRONG_GRAYSCALE)
    display_image(edges, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
