import numpy as np

def row_addition_elementary_matrix(n, target_row, source_row, scalar):
    """
    Creates an elementary matrix for the row addition operation.
    This matrix adds (scalar * source_row) to target_row.
    """
    E = np.identity(n)
    E[target_row, source_row] = scalar
    return E

def scalar_multiplication_elementary_matrix(n, row, scalar):
    """
    Creates an elementary matrix for the row scaling operation.
    This matrix scales the specified row by the given scalar.
    """
    E = np.identity(n)
    E[row, row] = scalar
    return E

"""
Function that finds the inverse of a non-singular matrix.
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""

def inverse(matrix):
    print(f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print("------------------------------------------------------------------------------------------------------------------")
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print("------------------------------------------------------------------------------------------------------------------")
                identity = np.dot(elementary_matrix, identity)

    return identity


if __name__ == '__main__':

    A = np.array([[1, -1, -2],
                  [2, -3, -5],
                  [-1, 3, 5]])

    try:
        A_inverse = inverse(A)
        print("\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================")

        import numpy as np

        A=np.array([[1, -1, -2], [2, -3, -5], [-1, 3, 5]])



        one_norm=np.linalg.norm(A, 1)
        print(f"||A||: {int(one_norm)}")


        norm_A_inv_1=np.linalg.norm(A_inverse, 1)
        print(f"The norm of the inverse of matrix A: {int(norm_A_inv_1)}")

        cond=one_norm*norm_A_inv_1
        print(f"COND = {int(cond)}")


    except ValueError as e:
        print(str(e))
