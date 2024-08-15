from scipy.sparse import coo_matrix  
import numpy as np 

import math as m

from helper import generate_dense_matrix_with_sparsity
print("***********************************************************************************")

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


sparse_matrix = generate_dense_matrix_with_sparsity(row_length_sparse, column_length_sparse, sparsity_sparse)
coo_form = coo_matrix(sparse_matrix)

dense_matrix = generate_dense_matrix_with_sparsity(
                row_length_dense, column_length_dense, sparsity_dense
            )


#dense_matrix = np.array([
    #[3, 8, 7, 1, 6, 5, 9, 4, 2, 10],
    #[5, 7, 9, 2, 8, 4, 1, 3, 6, 11],
    #[6, 1, 8, 3, 9, 2, 5, 7, 4, 12],
    #[7, 2, 5, 4, 10, 3, 6, 9, 8, 13],
    #[4, 9, 6, 5, 7, 1, 3, 8, 10, 14],
    #[8, 3, 4, 6, 11, 9, 2, 1, 7, 15],
    #[9, 4, 2, 7, 5, 8, 10, 6, 3, 16],
    #[2, 6, 3, 8, 4, 7, 11, 5, 9, 17],
    #[1, 5, 10, 9, 2, 6, 7, 3, 11, 18],
    #[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#])

#sparse_matrix = np.array([
    #[0, 0, 7, 0, 0, 5, 0, 0, 0, 0], 
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #[0, 0, 0, 0, 0, 0, 0, 4, 0, 0],  
    #[0, 0, 0, 0, 8, 0, 0, 0, 2, 0],  
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    #[0, 9, 0, 3, 0, 0, 0, 0, 0, 0],  
    #[0, 0, 1, 0, 0, 0, 0, 7, 0, 0], 
    #[0, 0, 0, 0, 0, 0, 0, 2, 0, 4],  
    #[1, 0, 0, 0, 0, 0, 0, 0, 5, 7],  
#])


coo_form = coo_matrix(sparse_matrix) 

DMCols = len(dense_matrix[0])

SPMrows = len(sparse_matrix)

coo_row = list(coo_form.row)
coo_col = list(coo_form.col)
coo_val = list(coo_form.data)

# print(coo_val)

numberOfAM = int(input("Enter the number of adders and multipliers: "))

adderDelay = int(input("Enter the adder delay: "))

multiplierDelay = int(input("Enter the multiplier delay: "))

total_clock_cycle = 0

scaling_factor = m.ceil(DMCols/numberOfAM)


for i in range(len(coo_val)):

    total_clock_cycle += (adderDelay + multiplierDelay)*scaling_factor 
    # print(total_clock_cycle)


print("The total time taken to compute the final matrix is ",total_clock_cycle)

print("***********************************************************************************")





