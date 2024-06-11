def bitmap_compression(matrix):
    # Flatten the matrix
    flat_list = [item for sublist in matrix for item in sublist]
    
    # Create bitmap and store non-zero values
    comp = []
    non_zero_values = []
    
    for value in flat_list:
        if value == 0:
            comp.append(0)
        else:
            comp.append(1)
            non_zero_values.append(value)
    
    # Calculate memory footprint
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    original_memory = rows * cols
    bitmap_memory = ((rows * cols + 7) // 8 ) + len(non_zero_values)+ rows + cols  # bits to bytes, round up
    comp_memory = bitmap_memory + len(non_zero_values)
    
    return comp_memory


