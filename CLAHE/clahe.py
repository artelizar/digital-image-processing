"""
Implementation of Adaptive Histogram Equalization (AHE) and
Contrast Limited Adaptive Histogram Equalization (CLAHE) methods for increasing contrast and image detail
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Parameters
NY = 8
NX = 8
L = 256
C = 80

SOURCE_PATH = "source/image.jpg"
RESULT_AHE_IMG_PATH = "result/image_after_ahe.png"
RESULT_CLAHE_IMG_PATH = "result/image_after_clahe.png"


def linear_interpolation(x, x1, y1, x2, y2):
    """
    Linear interpolation function
    """
    y = (y2 - y1) * (x - x1) / (x2 - x1) + y1

    return y


def bilinear_interpolation(cur_point, interpolation_points):
    """
    Bilinear interpolation function
    """
    y, x = cur_point
    y1, x1, v1 = interpolation_points[0]
    y2, x2, v2 = interpolation_points[1]
    y3, x3, v3 = interpolation_points[2]
    y4, x4, v4 = interpolation_points[3]

    r1_y, r1_x, r1_v = y, x1, linear_interpolation(y, y1, v1, y2, v2)
    r2_y, r2_x, r2_v = y, x3, linear_interpolation(y, y3, v3, y4, v4)
    v = linear_interpolation(x, r1_x, r1_v, r2_x, r2_v)

    return v


class CLAHE:
    """
    CLAHE algorithm class implementation
    """
    img = None
    my = None
    mx = None
    cdfs_matrix = None

    def __init__(self, Ny, Nx, L, C):
        self.Ny = Ny
        self.Nx = Nx
        self.L = L
        self.C = C

    def calculate_clipping_limit(self, hist):
        """
        Calculating an actual histogram clipping limit using binary search
        """
        top, bottom = self.C, 0
        while top - bottom > 1:
            middle = (top + bottom) // 2

            r = hist - middle
            s = r[r > 0].sum()
            if s > (self.C - middle) * self.L:
                top = middle
            else:
                bottom = middle

        return bottom

    def clip_hist(self, hist):
        """
        Histogram clipping function
        """
        # Calculate a histogram clipping limit
        p = self.calculate_clipping_limit(hist)
        # Calculate a residue
        r = hist - p
        indices = r > 0
        s = r[indices].sum()

        # Clip the histogram
        hist[indices] = p
        # Distribute the residue uniform
        hist += s // self.L
        hist[:s % self.L] += 1

    def hist_equalization(self, img, clip):
        """
        Image histogram equalization function
        :param clip: bool; if True, then clip image histogram
        """
        # Calculate image histogram
        values, counts = np.unique(img, return_counts=True)
        hist = np.zeros(self.L, dtype=np.int32)
        hist[values] = counts
        if clip:
            self.clip_hist(hist)

        # Calculate a histogram's cumulative distribution function (CDF)
        cdf = np.cumsum(hist, dtype=np.int64)
        # Normalize the histogram's CDF and bring it to the range [0; L)
        cdf = np.round(cdf / cdf.max() * (self.L - 1))
        cdf = cdf.astype(np.uint8)

        return cdf

    def create(self, img, clip=True):
        """
        Dividing an image into rectangles and calculating the histogram's CDF for each image rectangle
        """
        self.img = img
        self.my = img.shape[0] // self.Ny
        self.mx = img.shape[1] // self.Nx

        self.cdfs_matrix = np.empty((self.Ny, self.Nx, self.L), dtype=np.uint8)
        for i in range(self.Ny):
            for j in range(self.Nx):
                image_window = img[i * self.my: (i + 1) * self.my, j * self.mx: (j + 1) * self.mx]
                self.cdfs_matrix[i, j] = self.hist_equalization(image_window, clip=clip)

    def return_transformed_value(self, i, j, value):
        """
        Returns a new point value from the CDF of rectangle (i, j) and -1 for rectangles outside an image
        """
        if i < 0 or i >= self.Ny or j < 0 or j >= self.Nx:
            return -1
        else:
            return self.cdfs_matrix[i, j, value]

    def return_interpolated_value(self, cur_point, value, neighbour_positions):
        """
        Returns an interpolated pixel value for a current point
        """
        # Calculate new values of the current point from neighbour rectangle's CDFs
        transformed_values = []
        for i, j in neighbour_positions:
            transformed_values.append(self.return_transformed_value(i, j, value))
        transformed_values = np.array(transformed_values, dtype=np.int16)

        # Calculate coordinates of centers of the neighboring rectangles
        indices = transformed_values != -1
        neighbour_positions, transformed_values = neighbour_positions[indices], transformed_values[indices]
        interpolation_points = []
        for k, (i, j) in enumerate(neighbour_positions):
            interpolation_points.append((i * self.my + self.my // 2, j * self.mx + self.mx // 2, transformed_values[k]))

        # Interpolate the points depending on the number of the neighboring rectangles
        if len(interpolation_points) == 1:
            return interpolation_points[0][-1]
        elif len(interpolation_points) == 2:
            y, x = cur_point
            y1, x1, v1 = interpolation_points[0]
            y2, x2, v2 = interpolation_points[1]

            if x1 == x2:
                return linear_interpolation(y, y1, v1, y2, v2)
            else:
                return linear_interpolation(x, x1, v1, x2, v2)
        else:
            return bilinear_interpolation(cur_point, interpolation_points)

    def solve(self):
        """
        Applying the CLAHE algorithm to an image
        """
        result = np.empty_like(self.img)
        # Loop through image rectangles
        for i in range(self.Ny):
            for j in range(self.Nx):
                # Loop over an image rectangle
                for k in range(self.my):
                    for l in range(self.mx):
                        cur_point = (i * self.my + k, j * self.mx + l)

                        if k == self.my // 2 and l == self.mx // 2:
                            # Use a current rectangle
                            result[cur_point] = self.cdfs_matrix[i, j, self.img[cur_point]]
                        else:
                            # Find indices of nearest rectangles
                            di = -1 if k < self.my // 2 else 1
                            dj = -1 if l < self.mx // 2 else 1
                            neighbour_positions = np.array([(i, j), (i + di, j), (i, j + dj), (i + di, j + dj)],
                                                           dtype=np.int32)

                            # Calculate an interpolated pixel value for the current point
                            cur_point_value = self.img[cur_point]
                            result[cur_point] = self.return_interpolated_value(cur_point, cur_point_value,
                                                                               neighbour_positions)

        return result


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

    # Create a CLAHE class object
    clahe = CLAHE(Ny=NY, Nx=NX, L=L, C=C)

    # Display and save an image after the AHE algorithm (clip=False)
    clahe.create(img, clip=False)
    img_after_ahe = clahe.solve()
    display_image(
        img_after_ahe,
        title="Result image after AHE",
        save_path=os.path.join(*RESULT_AHE_IMG_PATH.split('/'))
    )

    # Display and save an image after the CLAHE algorithm
    clahe.create(img)
    img_after_clahe = clahe.solve()
    display_image(
        img_after_clahe,
        title="Result image after CLAHE",
        save_path=os.path.join(*RESULT_CLAHE_IMG_PATH.split('/'))
    )
    plt.show()


if __name__ == "__main__":
    main()
