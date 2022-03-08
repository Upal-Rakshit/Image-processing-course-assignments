# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def my_SEG(img):
    kernel_size = 39
    sigma = 6.33
    n = 50

    h, w = img.shape[:2]

    output_img = np.zeros([h, w], dtype=np.uint8)

    blue_img, green_img, red_img = cv2.split(img)

    k = int((kernel_size) / 2)

    kernel = np.ones((kernel_size, kernel_size))

    sigma_sq = sigma * sigma

    sum_ker = 0

    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            d_sq = (i * i) + (j * j)
            kernel[i + k, j + k] = (1 / (2 * math.pi * sigma_sq)) * math.exp(-(d_sq / (2 * sigma_sq)))
            sum_ker = sum_ker + kernel[i + k, j + k]
    kernel_norm = kernel / sum_ker

    blue_img_fil = cv2.filter2D(blue_img, -1, kernel_norm)
    green_img_fil = cv2.filter2D(green_img, -1, kernel_norm)
    red_img_fil = cv2.filter2D(red_img, -1, kernel_norm)

    out_img = cv2.merge([blue_img_fil, green_img_fil, red_img_fil])

    image = (0.299 * out_img[:, :, 2]) + (0.587 * out_img[:, :, 1]) + (0.114 * out_img[:, :, 0])

    data = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    _, labels, centers = cv2.kmeans(data, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    output_img[segmented_data >= 80] = 0
    output_img[segmented_data < 80] = 255

    return output_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    im = cv2.imread('iitk.jpg')
    im1 = my_SEG(im)

    cv2.imwrite('fordrone.jpg', im1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
