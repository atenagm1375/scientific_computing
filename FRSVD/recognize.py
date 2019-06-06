import numpy as np
import cv2


def load(names):
    vals = []
    for name in names:
        # exec('%s = %d' % (name, np.load(DIR + name + '.npy')))
        vars()[name] = np.load(DIR + name + '.npy')
        vals.append(vars()[name])
    return vals


def recognize(f, fbar, X, ur):
    f0 = f - fbar
    x = np.matmul(ur.transpose(), f0)
    fp = np.matmul(ur, x)
    D = X - x.reshape((x.shape[0], 1)) * np.ones((1, X.shape[1]))
    d = np.sqrt(np.diag(np.matmul(D.transpose(), D)))
    indx = np.argmin(d)
    return indx


def __main__():
    fbar, A, u, sigma, v = load(['fbar', 'A', 'U', 'Sigma', 'V'])
    x_test, y_test, y_train = load(['x_test', 'y_test', 'y_train'])
    r = 100
    ur = u[:, :r]
    X = np.matmul(ur.transpose(), A)
    count = 0
    for f, label in zip(x_test, y_test):
        indx = recognize(f, fbar, X, ur)
        print('This image is recognized as {}. It really is {}.'.format(y_train[indx], label))
        if y_train[indx] != label:
            count += 1
    print('error:', count / len(y_test))


DIR = './temp/'
__main__()
