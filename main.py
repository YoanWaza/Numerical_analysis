'''
גבריאל בנסמון
יואן הגיל
יואלי ברתל
'''


import numpy as np
from numpy.linalg import norm


def is_diagonally_dominant(A):
    abs_A=np.abs(A)
    diag=np.diag(abs_A)
    off_diag_sum=np.sum(abs_A, axis=1) - diag
    return np.all(diag >= off_diag_sum)


def gauss_seidel(A, b, X0, TOL=0.001, N=200):
    n=len(A)
    k=1

    if not is_diagonally_dominant(A):
        print('Matrix is not diagonally dominant.')
        return None

    print('Matrix is diagonally dominant - performing Gauss-Seidel algorithm\n')
    print(
        "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    x=np.zeros(n, dtype=np.double)
    while k <= N:
        for i in range(n):
            sigma=0
            for j in range(n):
                if j != i:
                    sigma+=A[i][j] * x[j]
            x[i]=(b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(round(val, 6)) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(np.round(x, 6).tolist())  # Round before returning

        k+=1
        X0=x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(np.round(x, 6).tolist())  # Round before returning


def jacobi_iterative(A, b, X0, TOL=0.001, N=200):
    n=len(A)
    k=1

    if not is_diagonally_dominant(A):
        print('Matrix is not diagonally dominant.')
        return None

    print('Matrix is diagonally dominant - performing Jacobi algorithm\n')
    print(
        "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= N:
        x=np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma=0
            for j in range(n):
                if j != i:
                    sigma+=A[i][j] * X0[j]
            x[i]=(b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(round(val, 6)) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(np.round(x, 6).tolist())  # Round before returning

        k+=1
        X0=x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(np.round(x, 6).tolist())  # Round before returning


if __name__ == "__main__":
    A=np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b=np.array([4, -1, -3])

    x=np.zeros_like(b, dtype=np.double)

    print("Gauss-Seidel Method:")
    gs_solution=gauss_seidel(A, b, x)
    if gs_solution:
        print("\nApproximate solution using Gauss-Seidel:", gs_solution)

    print("\nJacobi Method:")
    jacobi_solution=jacobi_iterative(A, b, x)
    if jacobi_solution:
        print("\nApproximate solution using Jacobi:", jacobi_solution)


