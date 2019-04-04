import cv2
import numpy as np
from scipy.fftpack import dct, idct
from constants import norm, filter, e


def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def normalize_image(img):
    return np.subtract(img, norm)


def denormalize_image(img):
    return np.round(np.add(img, norm))


def transform(block):
    axes = list(range(block.ndim))
    for ax in axes[::-1]:
        block = dct(block, type=2, axis=ax, norm='ortho')
    return block
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def inverse_transform(block):
    axes = list(range(block.ndim))
    for ax in axes[::-1]:
        block = idct(block, type=2, axis=ax, norm='ortho')
    return block
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def quantize(block):
    return np.round(np.divide(block, e))


def dequantize(block):
    return np.multiply(block, e)


def to_jpg(img):
    jpg_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], filter[0]):
        for j in range(0, img.shape[1], filter[1]):
            jpg_img[i : i + filter[0], j : j + filter[1]] = \
                        transform(img[i : i + filter[0], j : j + filter[1]])

            jpg_img[i : i + filter[0], j : j + filter[1]] = \
                        quantize(jpg_img[i : i + filter[0], j : j + filter[1]])

    return jpg_img


def from_jpg(img):
    iimg = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], filter[0]):
        for j in range(0, img.shape[1], filter[1]):
            iimg[i : i + filter[0], j : j + filter[1]] = \
                        dequantize(img[i : i + filter[0], j : j + filter[1]])
            print(np.sum(iimg[i : i + filter[0], j : j + filter[1]] > 0)/8/8)
            iimg[i : i + filter[0], j : j + filter[1]] = \
                        inverse_transform(iimg[i : i + filter[0], j : j + filter[1]])
            # print(iimg[i : i + filter[0], j : j + filter[1]])

    return iimg

img = cv2.imread("test.tif", 0)
show_image("original image", img)
img = normalize_image(img)
show_image("normalized image", img)
jpg_img = to_jpg(img)
print(np.min(np.min(jpg_img)), np.max(np.max(jpg_img)))
show_image("jpg image without compression", jpg_img)
iimg = denormalize_image(from_jpg(jpg_img))
print(np.min(np.min(iimg)), np.max(np.max(iimg)))
show_image("inverse image", iimg)
dif = img - iimg
show_image("dif", dif)
