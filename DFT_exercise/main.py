import numpy as np
import cv2


def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


img = cv2.imread("norm.jpg", 0)
show_image("curve", img)

Z = np.fft.fft2(img)
Zlog = np.log2(1 + np.abs(Z))
M = np.max(np.max(Zlog))
# print("Z:\n", Z)
# print("Zlog:\n", Zlog)
# print("M:\n", M)
show_image("dft", Zlog/M)

M = np.max(np.max(np.abs(Z)))
thresh_list = [0.0001, 0.001, 0.01, 0.05]

for thresh in thresh_list:
    Zthresh = np.multiply(np.abs(Z) > thresh * M, Z)
    print("Fraction of Fourier coefficients survived the cut:",
            np.sum(np.sum(np.abs(Zthresh) > 0)) / img.shape[0] / img.shape[1])

    zthresh = np.real(np.fft.ifft2(Zthresh))
    show_image("dft with thresholding", zthresh)

    print("Distortion of compressed signal:", 100 * np.linalg.norm(img - zthresh, 'fro') ** 2 \
            / np.linalg.norm(img, 'fro') ** 2)
