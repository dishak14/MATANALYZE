def run_length_encoding(matrix):
    # Flatten the matrix
    flat = [item for sublist in matrix for item in sublist]
    
    # Apply RLE
    comp = []
    prev_value = flat[0]
    count = 1
    
    for value in flat[1:]:
        if value == prev_value:
            count += 1
        else:
            comp.append((prev_value, count))
            prev_value = value
            count = 1
    
    # Don't forget to append the last set of values
    comp.append((prev_value, count))
    
    # Get the number of rows and columns
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    # Calculate memory footprint
    comp_memory = len(comp) * 2  # each (value, count) pair
    comp_memory += 2  # additional 2 integers for rows and cols
    
    return comp_memory

