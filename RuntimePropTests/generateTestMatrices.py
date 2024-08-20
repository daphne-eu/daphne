import numpy as np
import json

def generate_random_matrix(num_rows, num_cols, sparsity):
    size = num_rows * num_cols
    num_zeros = int(size * sparsity)
    num_non_zeros = size - num_zeros
    matrix_flat = np.zeros(size, dtype=np.int64)
    matrix_flat[:num_non_zeros] = np.random.randint(1, 11, size=num_non_zeros)
    np.random.shuffle(matrix_flat)
    matrix = matrix_flat.reshape((num_rows, num_cols))
    return matrix

def generate_sparsity_pattern_matrix(num_rows, num_cols, sparsity):
    total_elements = num_rows * num_cols

    num_zeros = int(total_elements * sparsity)
    num_non_zeros = total_elements - num_zeros

    full_zero_cols = num_zeros // num_rows
    remaining_zeros = num_zeros % num_rows

    matrix = np.zeros((num_rows, num_cols), dtype=np.int64)

    if remaining_zeros > 0:
        matrix[:remaining_zeros, full_zero_cols] = 0
        start_col = full_zero_cols + 1
    else:
        start_col = full_zero_cols

    if start_col < num_cols:
        non_zero_matrix_flat = np.random.randint(1, 11, size=num_non_zeros)
        matrix_flat = matrix.flatten()
        matrix_flat[num_zeros:] = non_zero_matrix_flat
        matrix = matrix_flat.reshape((num_rows, num_cols))

    return matrix

def flip_sparsity_and_nnz(matrix):
    flipped_matrix = np.zeros_like(matrix)
    non_zero_positions = matrix != 0

    flipped_matrix[:, :matrix.shape[1] - non_zero_positions.sum(axis=0).max()] = matrix[:, non_zero_positions.sum(axis=0).max():]
    flipped_matrix[:, matrix.shape[1] - non_zero_positions.sum(axis=0).max():] = 0

    return flipped_matrix

def save_matrix_to_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=',', fmt='%d')

def save_meta_file(matrix, filename):
    num_rows, num_cols = matrix.shape
    num_non_zeros = np.count_nonzero(matrix)

    meta_data = {
        "numRows": num_rows,
        "numCols": num_cols,
        "valueType": "f64",
        "numNonZeros": int(num_non_zeros)
    }

    with open(filename, 'w') as meta_file:
        json.dump(meta_data, meta_file, indent=4)

def generate_and_save_matrices(case):
    num_rows, num_cols = 5000, 5000
    sparsity = 0.4

    if case == 1:
        matrix1 = generate_sparsity_pattern_matrix(num_rows, num_cols, sparsity)
        matrix2 = generate_sparsity_pattern_matrix(num_rows, num_cols, sparsity)
        save_matrix_to_csv(matrix1, 'case1_matrix1.csv')
        save_matrix_to_csv(matrix2, 'case1_matrix2.csv')
        save_meta_file(matrix2, 'case1_matrix1.csv.meta')
        save_meta_file(matrix2, 'case1_matrix2.csv.meta')

    elif case == 2:
        matrix1 = generate_sparsity_pattern_matrix(num_rows, num_cols, sparsity)
        matrix2 = np.zeros((num_rows, num_cols), dtype=np.int64)
        mask_zeros = (matrix1 == 0)
        matrix2[mask_zeros] = np.random.randint(1, 11, size=np.count_nonzero(mask_zeros))

        save_matrix_to_csv(matrix1, 'case2_matrix1.csv')
        save_matrix_to_csv(matrix2, 'case2_matrix2.csv')
        save_meta_file(matrix1, 'case2_matrix1.csv.meta')
        save_meta_file(matrix2, 'case2_matrix2.csv.meta')

    elif case == 3:
        matrix1 = generate_random_matrix(num_rows, num_cols, sparsity)
        matrix2 = generate_random_matrix(num_rows, num_cols, sparsity)

        save_matrix_to_csv(matrix1, 'case3_matrix1.csv')
        save_matrix_to_csv(matrix2, 'case3_matrix2.csv')
        save_meta_file(matrix1, 'case3_matrix1.csv.meta')
        save_meta_file(matrix2, 'case3_matrix2.csv.meta')

    else:
        print("Invalid case number. Please use case 1,2,3.")

generate_and_save_matrices(1)
generate_and_save_matrices(2)
generate_and_save_matrices(3)

print("Matrices and meta files generated successfully.")