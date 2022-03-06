"""
Implementation of the document image skew correction algorithm
"""

import os
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


# Parameters
FC_THRESHOLD = 0.8
VOTES_THRESHOLD = 5

SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def find_angle(img, fc_threshold, votes_threshold):
    """
    Finding the angle of the correct orientation of a text on an image
    """
    # Compute the 2-dimensional discrete Fourier transform from the image
    fc = np.fft.fft2(img)
    # Shift the DC coefficient to the center of the spectrum
    fc = np.fft.fftshift(fc)
    # Transition to real Fourier coefficients
    fc = np.abs(fc)
    # Take the logarithm of Fourier coefficients
    fc = np.log(fc)
    # Normalize the Fourier coefficients
    fc -= fc.min()
    fc /= fc.max()

    # Suppress of Fourier coefficients using thresholding and binarization
    fc[fc < fc_threshold] = 0
    fc[fc != 0] = 1
    fc = fc.astype(np.uint8)

    # Apply the Hough transform to binarized Fourier coefficients and detect lines
    lines = cv2.HoughLines(fc, 1, np.pi / 180, threshold=votes_threshold).squeeze()
    # Find the angle of the correct text orientation
    angle = lines[0, 1] / np.pi * 180

    return angle


def edit_skew_correction(img, angle, cval=255):
    """
    Edit the orientation of a text on an image to the correct one using a found angle
    """
    rotated_img = ndimage.rotate(img, angle, reshape=False, cval=cval)

    return rotated_img


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

    # Find and display the correct orientation angle
    angle = find_angle(img, fc_threshold=FC_THRESHOLD, votes_threshold=VOTES_THRESHOLD)
    print("Angle =", round(angle))

    # Rotate the source image to the correct angle, display and save a result
    correct_img = edit_skew_correction(img, angle)
    display_image(correct_img, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
