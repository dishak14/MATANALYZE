from scipy.sparse import coo_matrix  
import numpy as np 

import math as m

from helper import generate_dense_matrix_with_sparsity

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


sparse_matrix = generate_dense_matrix_with_sparsity(
                row_length_sparse, column_length_sparse, sparsity_sparse
            )
coo_form = coo_matrix(sparse_matrix)

dense_matrix = generate_dense_matrix_with_sparsity(
                row_length_dense, column_length_dense, sparsity_dense
            )


DMCols = len(dense_matrix[0])
SPMrows = len(sparse_matrix)



coo_row = list(coo_form.row)
coo_col = list(coo_form.col)
coo_val = list(coo_form.data)

print(type(coo_row))

class unit:

    def __init__(self,totalDCol,numberOfAM = 1,adderDelay = 1,multiplierDelay = 1):

        self.AM = numberOfAM
        self.adderDelay = adderDelay
        self.multiplierDelay = multiplierDelay
        self.totalDCol = totalDCol

        self.RPE = m.ceil(self.totalDCol/self.AM)

        


        self.idk_dict = {}
        self.number = 1

        for i in range(self.AM):

            self.idk_dict["ADDER/MULTIPLIER" + str(self.number)] = []
            self.number += 1


    def operation(self,totalRow):
           
           self.TRPR = totalRow * self.RPE
            
           self.totalDelay = self.TRPR * (self.adderDelay + self.multiplierDelay)


           self.mod = self.totalDCol % self.AM
           self.number = self.AM

           return self.totalDelay


        #    for i in range(self.AM) :
        #        if(self.mod > 0):
        #             self.idk_dict["ADDER/MULTIPLIER" + self.number] = [self.RPE - 1,self.RPE]
        #        self.mod -= self.mod
        #        self.number -= 1

            # utilization  / element


numberOfUnits = int(input("Enter the total number of units: "))

numberOfAM = int(input("Enter the number of adders and multipliers in each unit: "))

adderDelay = int(input("Enter the adder delay: "))

multiplierDelay = int(input("Enter the multiplier delay: "))

hardware = {}
unit_number = 0
total_clock_cycle = 0

for i in range(numberOfUnits):
    hardware[unit_number] = unit(DMCols,numberOfAM,adderDelay,multiplierDelay)
    unit_number += 1

total_delay_of_all_units = []

for i in range(numberOfUnits) :
    total_delay_of_all_units.append(0)

print(total_delay_of_all_units)


for i in range(SPMrows):

    modulus = i % numberOfUnits
    cel = hardware[modulus]
    no = coo_row.count(i)

    total_delay_of_all_units[modulus] +=cel.operation(no)



print(total_delay_of_all_units)

print("The total time taken to compute the final matrix is",max(total_delay_of_all_units))










