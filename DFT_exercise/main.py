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
