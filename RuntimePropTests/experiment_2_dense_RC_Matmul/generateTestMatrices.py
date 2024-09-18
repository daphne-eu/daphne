import numpy as np
import json
import random

def generateCMatrix(N):
    Cmatrix = np.zeros((N, N))
    Cmatrix[:, 0] = np.random.randint(1,11,N)    
    return Cmatrix

def generateRMatrix(N):
    Rmatrix = np.zeros((N, N))
    Rmatrix[0, :] = np.random.randint(1,11,N) 
    return Rmatrix

def generate_random_matrix(num_rows, num_cols, sparsity):
    size = num_rows * num_cols
    num_zeros = int(size * sparsity)
    num_non_zeros = size - num_zeros
    matrix_flat = np.zeros(size, dtype=np.int64)
    matrix_flat[:num_non_zeros] = np.random.randint(1, 11, size=num_non_zeros)
    np.random.shuffle(matrix_flat)
    matrix = matrix_flat.reshape((num_rows, num_cols))
    return matrix


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
  
def save_matrix_to_csv(matrix, filename):
    np.savetxt(filename, matrix, delimiter=',', fmt='%d')
          
def generate_and_save_matrices():
    N = 5000
    
    C  = generateCMatrix(N)
    save_meta_file(C, 'Cmatrix.csv.meta')
    save_matrix_to_csv(C, 'Cmatrix.csv')
    
    R = generateRMatrix(N)
    save_meta_file(R, 'Rmatrix.csv.meta')
    save_matrix_to_csv(R, 'Rmatrix.csv')
    
    sparsity = 1 -0.29
    
    A_R = generate_random_matrix(N,N,sparsity)
    save_meta_file(A_R, 'A_Rmatrix.csv.meta')
    save_matrix_to_csv(A_R, 'A_Rmatrix.csv')
    
    A_C = generate_random_matrix(N,N,sparsity)
    save_meta_file(A_C, 'A_Cmatrix.csv.meta')
    save_matrix_to_csv(A_C, 'A_Cmatrix.csv')

    
generate_and_save_matrices()