import cv2
import os
import numpy as np
from scipy.linalg import svd


def load_dataset(name='grimace'):
    dir = './datasets/{}/'.format(name)
    people_names = os.listdir(dir)
    dataset = []
    for key in people_names:
        dataset += [(cv2.imread(dir + key + '/' + x, 0), key) for x in os.listdir(dir + key)]
    for i in range(len(dataset)):
        s1, s2 = dataset[i][0].shape
        dataset[i] = (cv2.resize(dataset[i][0], (s1//2, s2//2)), dataset[i][1])
    return dataset


def split_train_test(dataset, test_rate=0.2):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    np.random.shuffle(dataset)
    test_count = int(test_rate * len(dataset))
    test = dataset[:test_count]
    train = dataset[test_count:]
    for x, y in train:
        x_train.append(np.array(x).flatten())
        y_train.append(y)
    for x, y in test:
        x_test.append(np.array(x).flatten())
        y_test.append(y)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def svd_train(x):
    fbar = np.mean(x)
    A = x.transpose() - fbar * np.ones((1, x.shape[0]))
    u, sigma, v = svd(A)
    return fbar, A, u, sigma, v


def __main__():
    dataset = load_dataset()
    x_train, y_train, x_test, y_test = split_train_test(dataset)
    save_dir = './temp/'
    np.save(save_dir + 'x_train', x_train)
    np.save(save_dir + 'y_train', y_train)
    np.save(save_dir + 'x_test', x_test)
    np.save(save_dir + 'y_test', y_test)
    fbar, A, u, sigma, v = svd_train(x_train)
    np.save(save_dir + 'fbar', fbar)
    np.save(save_dir + 'A', A)
    np.save(save_dir + 'U', u)
    np.save(save_dir + 'Sigma', sigma)
    np.save(save_dir + 'V', v)


__main__()
