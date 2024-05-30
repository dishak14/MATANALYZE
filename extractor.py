# Import necessary libraries
import os
from io import BytesIO
import tarfile
import re
import shutil
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix
import numpy as np

# Define source and destination directories
source_dir = './mtx_files/'  # Directory containing .tar.gz files
destination_dir = './sparse_matrices/'  # Destination directory for extracted files
dense_dir = './dense_matrices/'  # Directory for dense matrices

# Flag to allow dense matrix generation
allowDenseMatrixGeneration = False

# Delete existing directories if they exist and create new ones
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
    os.makedirs(destination_dir)
if os.path.exists(dense_dir):
    shutil.rmtree(dense_dir)
    os.makedirs(dense_dir)

# Extract .tar.gz files in the source directory
for file_name in os.listdir(source_dir):
    if file_name.endswith(".tar.gz"):
        archive_path = os.path.join(source_dir, file_name)
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    tar.extract(member, destination_dir)
                    break  # Exit the inner loop after extracting the first match
        except Exception as e:
            print(f"Error processing {archive_path}: {e}")

# Function to move contents of subdirectories to the main directory and delete them
def move_subdir_contents_and_delete(main_dir):
    """Moves contents of subdirectories to the main directory and deletes them.

    Args:
        main_dir (str): Path to the main directory. """
    for root, subdirs, files in os.walk(main_dir):
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)

            # Move files to main directory
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                new_path = os.path.join(main_dir, filename)
                shutil.move(file_path, new_path)

            # Delete empty subdirectory
            os.rmdir(subdir_path)

# Execute the function to move contents of subdirectories in destination_dir to destination_dir
move_subdir_contents_and_delete(destination_dir)

# Function to generate a dense matrix
def generate_dense_matrix(rows, cols):
    non_zero_values = np.random.uniform(1.0, 100.0, size=(rows, cols))
    return non_zero_values

# Function to convert a 2D array to Matrix Market format and save it to a file
def mtx_converter(array, filename):
  """Converts a 2D array to Matrix Market format and saves it to a file.

  Args:
      array (list): The 2D array to be converted.
      filename (str): The name of the output file (including '.mtx' extension).
  """
  rows, cols = len(array), len(array[0])  # Get dimensions
  nnz = 0  # Count non-zero elements (optional for MM format)

  # Calculate number of non-zero elements (optional for MM format)
  for row in array:
    nnz += len([val for val in row if val != 0])

  with open(filename, 'w') as f:
    # Write header line 
    f.write(f"%%MatrixMarket matrix coordinate pattern {rows} {cols}\n")
    f.write(f"{rows} {cols} {nnz}\n")
    # Write data lines
    for i, row in enumerate(array):
      for j, value in enumerate(row):
        if value != 0:  # Write only non-zero elements (optional for MM format)
          f.write(f"{i + 1} {j + 1} {value}\n")

# If allowed, generate dense matrices
if allowDenseMatrixGeneration: 
    for mtx_file in os.listdir(destination_dir):
        file_path = f'{destination_dir}/{mtx_file}'
        sparse_matrix = mmread(file_path) 
        length = len(sparse_matrix.A)
        dense_matrix = generate_dense_matrix(length, length)
        if length == len(dense_matrix):
            file_name = f'{dense_dir}/{mtx_file}'
            mtx_converter(dense_matrix, file_name)
        else: 
            print("Invalid")
