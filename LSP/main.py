import numpy as np
import numpy.linalg as la


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
    remainder = la.norm(bbar[r+1:])
    return x, remainder


def solve_lsp_using_qr(A, b, n, r):
    pass


def __main__():
    m, n = read_matrix_dimension()
    A = read_matrix(m, n)
    b = read_vector(m)
    rank_A = la.matrix_rank(A)
    x, remainder = solve_lsp_using_svd(A, b, n, rank_A)
    print(x, remainder)


__main__()
