import scipy
from scipy.io import (
    mminfo,
    mmread,
    mmwrite,
)  # Import modules for matrix manipulation and file I/O
from scipy.sparse import coo_matrix  # Import module for sparse matrix operations
import numpy as np  # Import NumPy for numerical computations
from helper import (
    generate_dense_matrix,
    save_as_mtx,
    generate_dense_matrix_with_sparsity,
)  # Import functions from a helper file
import os
import matplotlib.pyplot as plt

names = []
multiplications_row = []
additions_row = []
multiplications_col = []
additions_col = []
compression_formats = []

type = int(input("Enter the type: "))

# Specify directories for source and destination matrices
if type == 1:
    source_sparse_matrices_dir = "./sparse_matrices/"
elif type == 2:
    source_sparse_matrices_dir = "./individual/"
elif type == 3:
    source_sparse_matrices_dir = "./manual/"
elif type == 4:
    source_sparse_matrices_dir = "./individual/"


destination_dense_matrices_dir = "./dense_matrices/"
saveGeneratedDenseMatrices = (
    False  # Flag to control whether to save generated dense matrices
)

# a = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0],
#               [0.0, 0.0, 2.0, 3.0, 0.0],
#               [0.0, 0.0, 0.0, 0.0, 0.0,],
#               [0.0, 4.0, 0.0, 0.0, 5.0]
# ])
# a = np.array(
#     [0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 2.0, 3.0]
# ])
# a = np.array([
#     [1.0, 0.0, 2.0],
#     [0.0, 0.0, 0.0],
#     [4.0, 0.0, 0.0]
# ])
# coo = coo_matrix(a)
# mmwrite("5_bus_col.mtx", coo)

# Iterate through files in the sparse matrices directory
# print("-------------------Row Wise-------------------")
for item in os.scandir(source_sparse_matrices_dir):
    if item.is_file():
        file_dir_name = (
            f"{source_sparse_matrices_dir}/{item.name}"  # Construct full file path
        )
        file_name = item.name  # Extract filename
        names.append(file_name)

        # Get information about the sparse matrix file

        info = mminfo(file_dir_name)

        # # Read the sparse matrix from the file and convert it to COO format

        sparse_matrix = mmread(file_dir_name).A
        coo_form = coo_matrix(sparse_matrix)
        row_length = info[0]
        column_length = info[1]
        info0 = info[0]
        info1 = info[1]
        dense_matrix = generate_dense_matrix_with_sparsity(
            row_length, column_length, 0.0
        )

        # Get row and column lengths (using the same value for both)

        # dense_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Create a dense matrix (example for demonstration)
        # This part could be replaced with code that generates a dense matrix based on info
        # or other criteria
        # dense_matrix = np.array([
        #     [1.0, 2.0, 3.0, 4.0, 5.0],
        #     [1.0, 2.0, 3.0, 4.0, 5.0],
        #     [1.0, 2.0, 3.0, 4.0, 5.0],
        #     [1.0, 2.0, 3.0, 4.0, 5.0],
        #     [1.0, 2.0, 3.0, 4.0, 5.0]
        # ])
        # dense_matrix = np.array(
        #     [
        #         [1.0, 2.0, 3.0, 4.0, 5.0],
        #         [1.0, 2.0, 3.0, 4.0, 5.0],
        #         [1.0, 2.0, 3.0, 4.0, 5.0],
        #         [1.0, 2.0, 3.0, 4.0, 5.0],
        #         [1.0, 2.0, 3.0, 4.0, 5.0],
        #     ]
        # )
        if type == 4:
            row_length_sparse = int(input("ENTER THE SPARSE ROW LENGTH: "))
            print()
            column_length_sparse = int(input("ENTER THE SPARSE COLUMN LENGTH: "))
            print()
            row_length_dense = int(input("ENTER THE DENSE ROW LENGTH: "))
            print()
            column_length_dense = int(input("ENTER THE DENSE COLUMN LENGTH: "))
            print()

            sparsity_sparse = float(input("ENTER THE SPARSITY OF THE SPARSE MATRIX: "))
            print()
            sparsity_dense = float(input("ENTER THE SPARSITY OF THE DENSE MATRIX: "))

            info0 = row_length_sparse

            info1 = column_length_sparse
              # column_length = random.randint(1,1000)
            sparse_matrix = generate_dense_matrix_with_sparsity(
                row_length_sparse, column_length_sparse, sparsity_sparse
            )
            coo_form = coo_matrix(sparse_matrix)

            dense_matrix = generate_dense_matrix_with_sparsity(
                row_length_dense, column_length_dense, sparsity_dense
            )

        # Optionally save the generated dense matrix
        if saveGeneratedDenseMatrices:
            file_path = f"{destination_dense_matrices_dir}{file_name}"
            save_as_mtx(dense_matrix, file_path)

        # Initialize counters for tracking operations
        additionCounterRow = 0
        multiplicationCounterRow = 0

        # Create a final matrix to store the result

        final_row_matrix = np.zeros((row_length, column_length))
        final_col_matrix = np.zeros((row_length, column_length))

        if type == 4:
            final_row_matrix = np.zeros((row_length_sparse, column_length_dense))
            final_col_matrix = np.zeros((row_length_sparse, column_length_dense))



        # Iterate through the non-zero elements of the sparse matrix in COO format
        previous_row = -1
        for i in range(len(coo_form.row)):
            row = coo_form.row[i]
            col = coo_form.col[i]
            value = sparse_matrix[row][col]

            # Perform element-wise multiplication and addition, optimizing for repeated rows
            if row == previous_row:
                final_row_matrix[row] += value * dense_matrix[col]
                additionCounterRow += len(dense_matrix[col])
                multiplicationCounterRow += len(dense_matrix[col])
            else:
                final_row_matrix[row] = (
                    value * dense_matrix[col]
                )  # Assign directly for non-repeated rows
                multiplicationCounterRow += len(dense_matrix[col])
            previous_row = row
        print("***********************")
        print(f"Name: {file_name.split('.')[0]} Dimensions: {info0} X {info1}")
        print()
        NNZ = np.count_nonzero(sparse_matrix)
        print("NNZ =", NNZ)
        print()
        spar = (info0 * info1 - NNZ) * 100 / (info0 * info1)
        print("Sparsity = ", spar, "%")
        print()
        # print(final_row_matrix)
        multiplications_row.append(multiplicationCounterRow)
        additions_row.append(additionCounterRow)
        print("Row Wise Algo:")
        print(
            "Multiplications: ",
            multiplicationCounterRow,
            "\nAdditions: ",
            additionCounterRow,
        )
        print()

        # Col Wise Algo
        additionCounterCol = 0
        multiplicationCounterCol = 0
        previous_col = -1
        for i in range(len(dense_matrix[0])):
            for j in range(len(dense_matrix)):
                col = i
                row = j
                value = dense_matrix[row][col]
                sparse_col = sparse_matrix[:, row]
                final_col_matrix[:, col] += value * sparse_col
                multiplicationCounterCol += len(sparse_matrix)
                # for index in range(len(sparse_col)):
                #     val = sparse_col[index]
                #     if val == 0.0:
                #         multiplicationCounterCol -= 1
                if col == previous_col:
                    additionCounterCol += len(sparse_matrix)
                previous_col = col
        print("Col Wise Algo:")
        multiplications_col.append(multiplicationCounterRow)
        additions_col.append(additionCounterCol)

        print(
            "Multiplications: ",
            multiplicationCounterRow,
            "\nAdditions: ",
            additionCounterCol,
        )

        print()

        if(type == 4):
            COO_FP  = 3*NNZ
            CSR_FP = 2*NNZ + (row_length_sparse+1) 
            CSC_FP = 2*NNZ + (column_length_sparse+1)
            print("MEMORY FOOTPRINT: ")
            print()
            print("COO FORMAT: ",COO_FP,"bytes")
            print()
            print("CSR FORMAT: ",CSR_FP,"bytes")
            print()
            print("CSC FORMAT",CSC_FP,"bytes")
            print()

        print("***********************")
        if type == 3:
            print("THE VALUE IS", len(sparse_matrix))
            print(sparse_matrix)
            print("\tX")
            print(dense_matrix)
            print("\t=")
            print(final_row_matrix)
            print()

for index, name in enumerate(names):
    names[index] = name.split(".")[0]

X_axis = np.arange(len(names))

plt.bar(X_axis - 0.2, multiplications_row, 0.4, label="Row Algo")
plt.bar(X_axis + 0.2, multiplications_col, 0.4, label="Col Algo")

plt.xticks(X_axis, names)
plt.xlabel("Matrices")
plt.ylabel("Multiplications")
plt.title("Multiplication Comparison")
plt.legend()
plt.show()

X_axis = np.arange(len(names))

plt.bar(X_axis - 0.2, additions_row, 0.4, label="Row Algo")
plt.bar(X_axis + 0.2, additions_col, 0.4, label="Col Algo")

plt.xticks(X_axis, names)
plt.xlabel("Matrices")
plt.ylabel("Additions")
plt.title("Addition Comparison")
plt.legend()
plt.show()

if(type == 4):
    plt.bar(0.2 ,[COO_FP],0.3, label = "COO")
    plt.bar(0.6 ,[CSR_FP],0.3, label = "CSR")
    plt.bar(1 ,[CSC_FP],0.3, label = "CSC")

    plt.xticks(X_axis, names)
    plt.xlabel("COMPRESSION FORMATS")
    plt.ylabel("SPACE OCCUPIED (in bytes)")
    plt.title("MEMORY FOOTPRINT")
    plt.legend()
    plt.show()




print("----------------------------------------------")