import numpy as np

def inner_product(dense_matrix, sparse_matrix, print_result=False):
    # sparse_matrix -> m x n
    # dense_matrix -> n x p
    m = len(sparse_matrix)
    n = len(sparse_matrix[0])
    p = len(dense_matrix[0])

    num_of_multiplications = m * n * p
    num_of_additions = m * n * p

    if print_result:
        print()
        print(f"----------Inner Product----------")
       
        print(f"Number of Multiplications: {num_of_multiplications}")
        print(f"Number of Additions: {num_of_additions}")
        print()

    result = {
        "num_of_multiplications": num_of_multiplications,
        "num_of_additions": num_of_additions,
    }

    return result


def outer_product(dense_matrix, sparse_matrix, print_result=False):
    # sparse_matrix -> m x n
    # dense matrix -> n x p
    m = len(sparse_matrix)
    n = len(sparse_matrix[0])
    p = len(dense_matrix[0])

    num_of_multiplications = m * p * n
    num_of_additions = m * p * (n - 1)

    if print_result:
        print()
        print(f"----------Outer Product----------")
       
        print(f"Number of Multiplications: {num_of_multiplications}")
        print(f"Number of Additions: {num_of_additions}")
        print()

    result = {
        "num_of_multiplications": num_of_multiplications,
        "num_of_additions": num_of_additions,
    }

    return result