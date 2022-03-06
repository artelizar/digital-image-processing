"""
Implementation of the Two-Pass Connected Component Labeling (CCL) algorithm
"""

import numpy as np
from disjoint_set import DisjointSet


# Biases for the top half of pixel’s neighbors
BIASES = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]


def first_pass(img, ds):
    """
    The first pass is to assign temporary labels and class equivalences
    """
    labeled_img = np.full_like(img, -1, dtype=np.int32)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j]:
                # Get the labels of adjacent pixels (4-connectivity)
                st = set()
                for yb, xb in BIASES:
                    if img[i + yb, j + xb] and labeled_img[i + yb, j + xb] != -1:
                        st.add(labeled_img[i + yb, j + xb])

                # Label the current pixel and save the class equivalences
                if len(st):
                    cur_label = min(st)  # the smallest label
                    labeled_img[i, j] = cur_label
                    for label in st:
                        cur_label = ds.union_sets(cur_label, label)
                else:
                    labeled_img[i, j] = ds.make_set()

    return labeled_img


def second_pass(img, labeled_img, ds):
    """
    The second pass is to replace each temporary label by the root label of its equivalence class
    """
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j]:
                labeled_img[i, j] = ds.find_set(labeled_img[i, j])


def connected_component_labeling(img):
    """
    Two-Pass Connected Component Labeling algorithm implementation using a disjoint-set
    """
    ds = DisjointSet()
    labeled_img = first_pass(img, ds)
    second_pass(img, labeled_img, ds)

    return labeled_img
