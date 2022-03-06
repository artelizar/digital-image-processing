"""
The GUI program for selecting an HSV range and highlighting image objects in the selected HSV range
to segment desired objects
"""

import os
import numpy as np
import cv2


L = 255

# Parameters
WINDOW_NAME = "Window"
SOURCE_PATH = "source/image.jpg"


def nothing(*args):
    """
    Empty function for cv2.createTrackbar()
    """
    pass


def main():
    """
    Main called function
    """
    # Create a window and trackbars
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("h_min", WINDOW_NAME, 0, L, nothing)
    cv2.createTrackbar("h_max", WINDOW_NAME, L, L, nothing)
    cv2.createTrackbar("s_min", WINDOW_NAME, 0, L, nothing)
    cv2.createTrackbar("s_max", WINDOW_NAME, L, L, nothing)
    cv2.createTrackbar("v_min", WINDOW_NAME, 0, L, nothing)
    cv2.createTrackbar("v_max", WINDOW_NAME, L, L, nothing)

    # Read a source image and convert it to HSV
    img = cv2.imread(os.path.join(*SOURCE_PATH.split('/')))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        # Read HSV range values from trackbars
        h_min, h_max = cv2.getTrackbarPos("h_min", WINDOW_NAME), cv2.getTrackbarPos("h_max", WINDOW_NAME)
        s_min, s_max = cv2.getTrackbarPos("s_min", WINDOW_NAME), cv2.getTrackbarPos("s_max", WINDOW_NAME)
        v_min, v_max = cv2.getTrackbarPos("v_min", WINDOW_NAME), cv2.getTrackbarPos("v_max", WINDOW_NAME)
        hsv_min, hsv_max = np.array((h_min, s_min, v_min), dtype=np.uint8), \
                           np.array((h_max, s_max, v_max), dtype=np.uint8)

        # Highlight image objects in the selected HSV range and display a binary image
        bin_img = cv2.inRange(hsv, hsv_min, hsv_max)
        cv2.imshow(WINDOW_NAME, bin_img)

        # Until the Esc key is pressed
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
