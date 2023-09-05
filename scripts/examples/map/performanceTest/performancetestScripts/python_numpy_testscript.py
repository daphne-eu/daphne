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

def sinus(matrix):
    return np.sin(matrix)

def cosinus(matrix):
    return np.cos(matrix)

def natural_logarithm(matrix):
    return np.log(matrix)

def exponential(matrix):
    return np.exp(matrix)

def polynomial(matrix):
    return 5*np.power(matrix,5) + 4*np.power(matrix,4) + 3*np.power(matrix,3) + 2*np.power(matrix,2) + matrix

def relu(matrix):
    return np.maximum(0, matrix)

def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))

def hyperbolic_tangent(matrix):
    return np.tanh(matrix)

def thresholding(matrix):
    return np.where(matrix > 40, 10, 0)

def generate_matrix(dtype, matrix_size, min_value, max_value):
    if dtype in ["float64", "float32"]:
        return np.random.uniform(min_value, max_value, (matrix_size, matrix_size)).astype(dtype)
    else:
        return np.random.randint(min_value, max_value + 1, (matrix_size, matrix_size), dtype=dtype)

def main():
    if len(sys.argv) != 6:
        print("Usage: script.py <dtype> <operation> <matrix_size> <min_value> <max_value>")
        sys.exit(1)

    type_mapping = {
        "double": "float64",
        "float": "float32",
        "int64_t": "int64",
        "int32_t": "int32",
        "int8_t": "int8",
        "uint64_t": "uint64",
        "uint8_t": "uint8"
    }

    dtype = type_mapping.get(sys.argv[1])
    if not dtype:
        print("Invalid dtype. Choose from: double, float, int64_t, int32_t, int8_t, uint64_t, uint8_t.")
        sys.exit(1)

    operation = int(sys.argv[2])
    matrix_size = int(sys.argv[3])
    min_value = float(sys.argv[4])
    max_value = float(sys.argv[5])

    np.random.seed(42)
    matrix = generate_matrix(dtype, matrix_size, min_value, max_value)

    #print("Generated matrix:\n", matrix)
    
    if operation == 1:
        result = multiplication(matrix)
    elif operation == 2:
        result = power(matrix)
    elif operation == 3:
        result = sinus(matrix)
    elif operation == 4:
        result = cosinus(matrix)
    elif operation == 5:
        result = natural_logarithm(matrix)
    elif operation == 6:
        result = exponential(matrix)
    elif operation == 7:
        result = polynomial(matrix)
    elif operation == 8:
        result = relu(matrix)
    elif operation == 9:
        result = sigmoid(matrix)
    elif operation == 10:
        result = hyperbolic_tangent(matrix)
    elif operation == 11:
        result = thresholding(matrix)
    else:
        print("Invalid operation value")
        sys.exit(1)

    #print("\nResultant matrix after operation:\n", result)

if __name__ == '__main__':
    main()