from sklearn.datasets.samples_generator import make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd, eig


def load_dataset(name, n_samples, noise):
    if name == 'circles':
        dataset = make_circles(n_samples=n_samples, factor=.2, noise=noise)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=dataset[1])
    plt.show()
    return dataset


def construct_similarity_graph(x):
    W = kneighbors_graph(x, 25).toarray()
    return W


def compute_unnormalized_laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def compute_new_points(L, k):
    u, sigma, v = svd(L)
    uk = u[:, -k:]
    return uk


def cluster(y, k):
    kmeans = KMeans(k, random_state=0).fit(y)
    return kmeans.labels_


def __main__():
    n_samples = 500
    x, _ = load_dataset('circles', n_samples, 0.1)
    k = 2
    W = construct_similarity_graph(x)
    L = compute_unnormalized_laplacian(W)
    y = compute_new_points(L, k)
    C = cluster(y, k)
    plt.scatter(x[:, 0], x[:, 1], c=C)
    plt.show()


__main__()
