import cv2
import numpy as np
from scipy.fftpack import dct, idct
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from constants import norm, filter, e, img_size


def normalize_image(img):
    return img - norm


def denormalize_image(img):
    return img + norm


def transform(block):
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def inverse_transform(block):
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def quantize(block):
    return np.round(block / e)


def dequantize(block):
    return block * e


def to_jpg(img):
    img = normalize_image(img)
    jpg_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], filter[0]):
        for j in range(0, img.shape[1], filter[1]):
            jpg_img[i: i + filter[0], j: j + filter[1]] = \
                        transform(img[i: i + filter[0], j: j + filter[1]])

            jpg_img[i: i + filter[0], j: j + filter[1]] = quantize(jpg_img[i: i + filter[0], j: j + filter[1]])

    return jpg_img


def from_jpg(img):
    iimg = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], filter[0]):
        for j in range(0, img.shape[1], filter[1]):
            iimg[i: i + filter[0], j: j + filter[1]] = \
                        dequantize(img[i: i + filter[0], j: j + filter[1]])
            iimg[i: i + filter[0], j: j + filter[1]] = inverse_transform(iimg[i: i + filter[0], j: j + filter[1]])

    return denormalize_image(iimg)


Tk().withdraw()
file_name = askopenfilename(title="Choose an image file with size {}x{}".format(img_size, img_size))

image = np.zeros((img_size, img_size))
image[:img_size, :img_size] = cv2.imread(file_name, 0)
if image.shape != (img_size, img_size):
    raise Exception("Invalid image size")
jpg_image = to_jpg(image)

jpg_image = from_jpg(jpg_image)
cv2.imwrite("my_jpg.jpg", jpg_image)
