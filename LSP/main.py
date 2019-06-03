import numpy as np
import scipy.linalg as la
import time


def read_matrix_dimension():
    print("Enter dimension of matrix A(m >= n):", end=' ')
    try:
        m, n = (int(i) for i in input().strip().split())
        if m < n:
            print("Number of rows cannot be less than number of columns in LSP problem!")
            raise Exception()
    except Exception:
        print("Invalid input for matrix dimension. Try again by entering two space-separated integers:")
        return read_matrix_dimension()
    return m, n


def read_matrix(m, n):
    A = np.zeros((m, n))
    print("Enter rows of the matrix A(separate elements of each row with space):")
    try:
        for i in range(m):
            A[i, :] = [float(j) for j in input().strip().split()]
    except Exception:
        print("Invalid input for matrix A. Please try again.")
        return read_matrix(m, n)
    return A


def read_vector(m):
    b = np.zeros(m)
    print("Enter elements of the vector b of size {}(separate them with space):".format(m))
    try:
        b[:] = [float(j) for j in input().strip().split()]
    except Exception:
        print("Invalid input for vector b. Please try again!")
        return read_vector(m)
    return b


def solve_lsp_using_svd(A, b, n, r):
    u, sigma, vt = la.svd(A)
    bbar = np.transpose(u).dot(b)
    y = bbar[:r] / sigma[:r]
    if len(y) == 1:
        y = float(y)
    v = np.transpose(vt)[:, :r]
    x = v.dot(y)
    remainder = la.norm(bbar[r:]) if r < n else 0
    return x.reshape((n,)), remainder


def solve_lsp_using_qr(A, b, n, r):
    q, R, p = la.qr(A, pivoting=True)
    bbar = np.transpose(q).dot(b)
    y = la.lstsq(R[:n, :n], bbar[:n])[0]
    x = y[p]
    remainder = la.norm(bbar[r:]) if r < n else 0
    return x.reshape((n,)), remainder


def lsp_solver(func, A, b, n, r, method):
    start = time.time()
    x, remainder = func(A, b, n, r)
    svd_time = time.time() - start
    print("Solved LSP using {} in {} ms. Here's the results:".format(method, np.round(svd_time * 1000, decimals=2)))
    print("x:", x)
    print("remainder:", remainder)


def __main__():
    m, n = read_matrix_dimension()
    A = read_matrix(m, n)
    b = read_vector(m)
    rank_A = np.linalg.matrix_rank(A)
    lsp_solver(solve_lsp_using_svd, A, b, n, rank_A, 'SVD')
    lsp_solver(solve_lsp_using_qr, A, b, n, rank_A, 'QR')


__main__()
