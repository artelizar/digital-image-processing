"""
Implementation of the Suzuki's algorithm for finding contours and their hierarchical relationships of image objects
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


CLOCKWISE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
COUNTERCLOCKWISE = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
L = 255

# Parameters
ONLY_MOST_OUTER = True
W_MIN = 250
W_MAX = 500
H_MIN = 250
H_MAX = 500

CONTOURS_COLOR = (0, 0, 255)
CONTOURS_THICKNESS = 2
BOUNDING_BOXES_COLOR = (0, 0, 255)
BOUNDING_BOXES_THICKNESS = 3

# For selecting dark green objects
HSV_MIN_VALUES = (35, 100, 0)
HSV_MAX_VALUES = (80, 230, 130)

SOURCE_PATH = "source/image.jpg"
RESULT_PATH = "result/image.png"


def find_next_point(f, bp_i, bp_j, sp_i, sp_j, traversal, clockwise=False):
    """
    Finding the starting point (i1, j1) around a base point (i, j) traversing a contour clockwise
    or the next point (i4, j4) around a base point (i3, j3) traversing a contour counterclockwise
    """
    bias, idx = (sp_i - bp_i, sp_j - bp_j), 0
    for k in range(len(traversal)):
        if bias == traversal[k]:
            idx = k
            break
    # Start from the next point of the point (i2, j2) in counterclockwise order
    if not clockwise:
        idx = (idx + 1) % len(traversal)

    np_i, np_j, is_examined = -1, -1, False
    for k in range(len(traversal)):
        cur_bias = traversal[(idx + k) % len(traversal)]
        # Determine if the point (i3, j3 + 1) has been visited when traversing the contour counterclockwise
        if not clockwise and cur_bias == (0, 1):
            is_examined = True

        # Find a first nonzero point
        if f[bp_i + cur_bias[0], bp_j + cur_bias[1]]:
            np_i, np_j = bp_i + cur_bias[0], bp_j + cur_bias[1]
            break

    return np_i, np_j, is_examined


def contour_traversal(f, i, j, i2, j2, NBD):
    """
    The function of traversing and saving a found contour
    """
    contour = []

    # Find the starting point (i1, j1) of traversing the contour clockwise
    i1, j1, _ = find_next_point(f, i, j, i2, j2, traversal=CLOCKWISE, clockwise=True)
    if i1 == -1 and j1 == -1:
        f[i, j] = -NBD
        contour.append([j, i])  # column index first, then row index
    else:
        i2, j2 = i1, j1
        i3, j3 = i, j

        # Traverse the contour counterclockwise
        while True:
            contour.append([j3, i3])  # column index first, then row index

            i4, j4, is_examined = find_next_point(f, i3, j3, i2, j2, traversal=COUNTERCLOCKWISE)
            if is_examined and f[i3, j3 + 1] == 0:
                f[i3, j3] = -NBD
            elif not(is_examined and f[i3, j3 + 1] == 0) and f[i3, j3] == 1:
                f[i3, j3] = NBD

            # Condition for coming back to the starting point
            if (i4, j4) == (i, j) and (i3, j3) == (i1, j1):
                break
            else:
                i2, j2 = i3, j3
                i3, j3 = i4, j4

    return np.array(contour, dtype=np.int32)


def find_contours(bin_img):
    """
    Suzuki's algorithm implementation
    """
    contours = []  # list of contour point coordinates
    hierarchy = [['hb', -1]]  # list of values for each border: border type and parent border NBD
    NBD = 1

    # Add the frame of zeros to an image
    f = np.zeros((bin_img.shape[0] + 2, bin_img.shape[1] + 2), dtype=np.int64)
    f[1:f.shape[0] - 1, 1:f.shape[1] - 1] = bin_img

    # Loop through image pixels
    for i in range(1, f.shape[0] - 1):
        LNBD = 1
        for j in range(1, f.shape[1] - 2):
            if f[i, j] == 0 and f[i, j + 1] == 1:
                NBD += 1

                # Determine a parent for a current contour (ob' - outer border, 'hb' - hole border)
                if hierarchy[LNBD - 1][0] == 'hb':
                    hierarchy.append(['ob', LNBD])
                else:
                    hierarchy.append(['ob', hierarchy[LNBD - 1][1]])

                # Traverse and save the found contour
                contour = contour_traversal(f, i, j + 1, i, j, NBD)
                # -1 due to adding the frame of zeros to the image
                contours.append(contour - 1)

                if f[i, j + 1] != 1:
                    LNBD = abs(f[i, j + 1])
            elif f[i, j] >= 1 and f[i, j + 1] == 0:
                NBD += 1
                if f[i, j] > 1:
                    LNBD = f[i, j]

                # Determine a parent for a current contour (ob' - outer border, 'hb' - hole border)
                if hierarchy[LNBD - 1][0] == 'ob':
                    hierarchy.append(['hb', LNBD])
                else:
                    hierarchy.append(['hb', hierarchy[LNBD - 1][1]])

                # Traverse and save the found contour
                contour = contour_traversal(f, i, j, i, j + 1, NBD)
                # -1 due to adding the frame of zeros to the image
                contours.append(contour - 1)

                if f[i, j] != 1:
                    LNBD = abs(f[i, j])
            elif f[i, j] != 0 and f[i, j] != 1:
                LNBD = abs(f[i, j])

    return contours, hierarchy, f[1:f.shape[0] - 1, 1:f.shape[1] - 1]


def filter_contours(contours, hierarchy, only_most_outer=True):
    """
    Contours filtering function
    :param only_most_outer: bool; if True, then return only the outermost contours whose parent is frame,
    otherwise - return all contours
    """
    if only_most_outer:
        filtered_contours = []
        for i in range(1, len(hierarchy)):
            if hierarchy[i][1] == 1:
                filtered_contours.append(contours[i - 1])

        return filtered_contours
    else:
        return contours


def draw_bounding_boxes(img, contours, w_min, w_max, h_min, h_max, color='r', thickness=3):
    """
    Filtering the sizes of found objects and drawing their bounding boxes
    """
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w_min <= w <= w_max and h_min <= h <= h_max:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness)


def display_image(img, cmap=None, title="", save_path=None):
    """
    Image displaying and saving function
    """
    if cmap is None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)

    if save_path is not None:
        plt.imsave(save_path, img, cmap=cmap)


def main():
    """
    Main called function
    """
    # Read and display a source image
    img = cv2.imread(os.path.join(*SOURCE_PATH.split('/')))
    display_image(img, title="Source image")

    # Convert the source image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Filter the HSV image by the parameters found in find_hsv.py and binarize it
    hsv_min, hsv_max = np.array(HSV_MIN_VALUES, dtype=np.uint8), np.array(HSV_MAX_VALUES, dtype=np.uint8)
    bin_hsv = cv2.inRange(hsv, hsv_min, hsv_max)
    # Display the binary image
    display_image(bin_hsv, title="Binary image", cmap='gray')

    # Solve the problem using the Suzuki's algorithm
    bin_hsv = np.array(bin_hsv // L, dtype=np.int64)
    contours, hierarchy, _ = find_contours(bin_hsv)

    # Leave only the outermost contours
    contours = filter_contours(contours, hierarchy, only_most_outer=ONLY_MOST_OUTER)
    # Draw the contours
    cv2.drawContours(img, contours, -1, color=CONTOURS_COLOR, thickness=CONTOURS_THICKNESS)
    # Display the contours in the image
    display_image(img, title="Image with contours")

    # Draw bounding boxes, display and save a final image
    draw_bounding_boxes(img, contours, w_min=W_MIN, w_max=W_MAX, h_min=H_MIN, h_max=H_MAX,
                        color=BOUNDING_BOXES_COLOR, thickness=BOUNDING_BOXES_THICKNESS)
    display_image(img, title="Result image", save_path=os.path.join(*RESULT_PATH.split('/')))
    plt.show()


if __name__ == "__main__":
    main()
