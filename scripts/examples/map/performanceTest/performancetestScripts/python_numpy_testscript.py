# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright 2022 The DAPHNE Consortium
#
# -------------------------------------------------------------

import numpy as np
import sys

def multiplication(matrix):
    return matrix * 2

def power(matrix):
    return np.power(matrix, 3)

def logarithm_base_10(matrix):
    return np.log10(matrix)

def exponential(matrix):
    return np.exp(matrix**2)

def polynomial(matrix):
    return 5*np.power(matrix,5) + 4*np.power(matrix,4) + 3*np.power(matrix,3) + 2*np.power(matrix,2) + matrix

def polynomial3(matrix):
    return 3*np.power(matrix,3) + 2*np.power(matrix,2) + matrix

def relu(matrix):
    return np.maximum(0, matrix)

def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def thresholding(matrix):
    return np.where(matrix > 42, 10, 0)

def fibonacci(matrix):
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    return np.round((phi**matrix - psi**matrix) / np.sqrt(5))

def generate_matrix(dtype, matrix_size, min_value, max_value):
    if(dtype == np.float64 or dtype == np.float32):
        return np.random.uniform(min_value, max_value, (matrix_size, matrix_size)).astype(dtype)
    else:
        return np.random.randint(min_value, max_value + 1, (matrix_size, matrix_size), dtype=dtype)

def main():
    if len(sys.argv) != 6:
        print("Usage: script.py <dtype> <operation> <matrix_size> <min_value> <max_value>")
        sys.exit(1)

    numpy_type_map = {
    'f64': np.float64,
    'f32': np.float32,
    'int64': np.int64,
    'int32': np.int32,
    'int8': np.int8,
    'uint64': np.uint64,
    'uint8': np.uint8
    }
    
    dtype = numpy_type_map[sys.argv[1]]
    operation = int(sys.argv[2])
    matrix_size = int(sys.argv[3])
    min_value = float(sys.argv[4])
    max_value = float(sys.argv[5])

    np.random.seed(42)
    matrix = generate_matrix(dtype, matrix_size, min_value, max_value)
    
    if operation == 1:
        result = multiplication(matrix)
    elif operation == 2:
        result = power(matrix)
    elif operation == 3:
        result = logarithm_base_10(matrix)
    elif operation == 4:
        result = exponential(matrix)
    elif operation == 5:
        result = polynomial(matrix)
    elif operation == 6:
        result = relu(matrix)
    elif operation == 7:
        result = sigmoid(matrix)
    elif operation == 8:
        result = thresholding(matrix)
    elif operation == 9:
        result = fibonacci(matrix)
    elif operation == 10:
        result = polynomial3(matrix)
    else:
        print("Invalid operation value")
        sys.exit(1)

    #print("\nResultant matrix after operation:\n", result)

if __name__ == '__main__':
    main()