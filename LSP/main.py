import numpy as np


def read_matrix_dimension():
    print("Enter dimension of matrix A:", end=' ')
    try:
        m, n = (int(i) for i in input().strip().split())
    except Exception:
        print("Invalid input for matrix dimension. Try again by entering two space-separated integers:")
        return read_matrix_dimension()
    return m, n


def read_matrix(m, n):
    A = np.zeros((m, n))
    print("Enter rows of the matrix(separate elements of each row with space):")
    try:
        for i in range(m):
            A[i, :] = [float(j) for j in input().strip().split()]
    except Exception:
        print("Invalid input for a matrix. Please try again.")
        return read_matrix(m, n)
    return A


def solve_lsp_using_svd(A, m, n):
    pass


def solve_lsp_using_qr(A, m, n):
    pass


def __main__():
    m, n = read_matrix_dimension()
    A = read_matrix(m, n)


__main__()
