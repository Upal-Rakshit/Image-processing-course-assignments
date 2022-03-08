# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import compare_ssim as ssim
import time


def rt_BF(img):

    kernel_size = 11
    sigmad = 20
    sigmai = 30
    h, w = img.shape[:2]

    JK_blue = np.zeros((h, w), dtype=np.uint8)
    Jl_blue = np.zeros((h, w), dtype=np.uint8)
    JK_green = np.zeros((h, w), dtype=np.uint8)
    Jl_green = np.zeros((h, w), dtype=np.uint8)
    JK_red = np.zeros((h, w), dtype=np.uint8)
    Jl_red = np.zeros((h, w), dtype=np.uint8)

    blue_img, green_img, red_img = cv2.split(img)

    kernel = ((1 / kernel_size) ** 2) * np.ones((kernel_size, kernel_size))

    sigmad_sq = 2 * sigmad * sigmad

    sigmai_sq = 2 * sigmai * sigmai

    ker = int((kernel_size) / 2)
    kernel = np.ones((kernel_size, kernel_size))

    for i in range(-ker, ker + 1):
        for j in range(-ker, ker + 1):
            d_sq = (i * i) + (j * j)
            kernel[i + ker, j + ker] = (1 / (math.pi * sigmad_sq)) * math.exp(-(d_sq / sigmad_sq))

    k = np.arange(0, 256, 15)
    for p in range(len(k) - 1):
        JK_blue[blue_img >= k[p]] = k[p]
        Jl_blue[blue_img >= k[p]] = k[p + 1]
        JK_green[green_img >= k[p]] = k[p]
        Jl_green[green_img >= k[p]] = k[p + 1]
        JK_red[red_img >= k[p]] = k[p]
        Jl_red[red_img >= k[p]] = k[p + 1]

    q = (blue_img - JK_blue) / (Jl_blue - JK_blue)
    WK = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((JK_blue - blue_img) / sigmai) ** 2) * 0.5)
    Wl = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((Jl_blue - blue_img) / sigmai) ** 2) * 0.5)
    JK_blue = np.multiply(WK, blue_img)
    Jl_blue = np.multiply(Wl, blue_img)
    a1 = cv2.filter2D(JK_blue, -1, kernel)
    b1 = cv2.filter2D(WK, -1, kernel)
    a2 = cv2.filter2D(Jl_blue, -1, kernel)
    b2 = cv2.filter2D(Wl, -1, kernel)
    JK_B = np.true_divide(a1, b1, out=np.zeros_like(a1), where=b1 != 0)
    Jl_B = np.true_divide(a2, b2, out=np.zeros_like(a2), where=b2 != 0)
    blue_img_fil = np.multiply((1 - q), JK_B) + np.multiply(q, Jl_B)
    blue_img_fil = blue_img_fil.astype(np.uint8)

    q = (green_img - JK_green) / (Jl_green - JK_green)
    WK = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((JK_green - green_img) / sigmai) ** 2) * 0.5)
    Wl = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((Jl_green - green_img) / sigmai) ** 2) * 0.5)
    JK_green = np.multiply(WK, green_img)
    Jl_green = np.multiply(Wl, green_img)
    a1 = cv2.filter2D(JK_green, -1, kernel)
    b1 = cv2.filter2D(WK, -1, kernel)
    a2 = cv2.filter2D(Jl_green, -1, kernel)
    b2 = cv2.filter2D(Wl, -1, kernel)
    JK_B = np.true_divide(a1, b1, out=np.zeros_like(a1), where=b1 != 0)
    Jl_B = np.true_divide(a2, b2, out=np.zeros_like(a2), where=b2 != 0)
    green_img_fil = np.multiply((1 - q), JK_B) + np.multiply(q, Jl_B)
    green_img_fil = green_img_fil.astype(np.uint8)

    q = (red_img - JK_red) / (Jl_red - JK_red)
    WK = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((JK_red - red_img) / sigmai) ** 2) * 0.5)
    Wl = (1 / math.sqrt(math.pi * sigmai_sq)) * np.exp(-(((Jl_red - red_img) / sigmai) ** 2) * 0.5)
    JK_red = np.multiply(WK, red_img)
    Jl_red = np.multiply(Wl, red_img)
    a1 = cv2.filter2D(JK_red, -1, kernel)
    b1 = cv2.filter2D(WK, -1, kernel)
    a2 = cv2.filter2D(Jl_red, -1, kernel)
    b2 = cv2.filter2D(Wl, -1, kernel)
    JK_B = np.true_divide(a1, b1, out=np.zeros_like(a1), where=b1 != 0)
    Jl_B = np.true_divide(a2, b2, out=np.zeros_like(a2), where=b2 != 0)
    red_img_fil = np.multiply((1 - q), JK_B) + np.multiply(q, Jl_B)
    red_img_fil = red_img_fil.astype(np.uint8)

    out_img = cv2.merge([blue_img_fil, green_img_fil, red_img_fil])

    #############################
    # End your code here ########
    #############################

    return out_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    im = cv2.imread('iitk.jpg')
    im1 = rt_BF(im)
    cv2.imwrite('Denoised_iitk.jpg',im1)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
