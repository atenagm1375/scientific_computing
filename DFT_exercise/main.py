import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_image(name, img):
    print("Press any key to close the window(do not use x button on top of the window)")
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


zmg = cv2.imread("norm.jpg", 0)
img = np.zeros(zmg.shape)
img[:zmg.shape[0], :zmg.shape[1]] = zmg
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
compression_ratios = []
distortions = []

for thresh in thresh_list:
    Zthresh = np.multiply(np.abs(Z) > thresh * M, Z)

    compression_ratio = np.sum(np.sum(np.abs(Zthresh) > 0)) / img.shape[0] / img.shape[1]
    print("Fraction of Fourier coefficients survived the cut:",
            compression_ratio)
    compression_ratios.append(compression_ratio)

    zthresh = np.real(np.fft.ifft2(Zthresh))
    show_image("dft with thresholding", zthresh)

    distortion = 100 * np.linalg.norm(img - zthresh, 'fro') ** 2 / np.linalg.norm(img, 'fro') ** 2
    print("Distortion of compressed signal:", distortion)
    distortions.append(distortion)

plt.plot(distortions, compression_ratios)
plt.xlabel(xlabel="distortion")
plt.ylabel(ylabel="compression ratio")
plt.show()
