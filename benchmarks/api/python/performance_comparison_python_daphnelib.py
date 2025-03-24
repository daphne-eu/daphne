from cProfile import label

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from daphne.context import DaphneContext
import tensorflow as tf
import torch

# Initialize Daphne context
dctx = DaphneContext()

# Define number of repetitions for averaging
repetitions = 10

# Results dictionary
results = {
    "operation": [],
    "size": [],
    "numpy": [],
    "tensorflow": [],
    "pandas": [],
    "pytorch": [],
    "daphnelib": []
}

# Benchmark scenarios
def benchmark_operation(operation_name, size, numpy_func, tensorflow_func, pandas_func, torch_func, daphnelib_func):
    # Generate random data
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # NumPy
    numpy_times = []
    for _ in range(repetitions):
        start = time.time()
        numpy_func(A, B)
        end = time.time()
        numpy_times.append(end - start)
    numpy_avg = np.mean(numpy_times)

    # TensorFlow
    tf_A = tf.constant(A)
    tf_B = tf.constant(B)
    tensorflow_times = []
    for _ in range(repetitions):
        start = time.time()
        tensorflow_func(tf_A, tf_B)
        end = time.time()
        tensorflow_times.append(end - start)
    tensorflow_avg = np.mean(tensorflow_times)

    # Pandas
    pandas_A = pd.DataFrame(A)
    pandas_B = pd.DataFrame(B)
    pandas_times = []
    for _ in range(repetitions):
        start = time.time()
        pandas_func(pandas_A, pandas_B)
        end = time.time()
        pandas_times.append(end - start)
    pandas_avg = np.mean(pandas_times)

    # Pytorch
    torch_A = torch.from_numpy(A)
    torch_B = torch.from_numpy(B)
    torch_times = []
    for _ in range(repetitions):
        start = time.time()
        torch_func(torch_A, torch_B)
        end = time.time()
        torch_times.append(end - start)
    torch_avg = np.mean(torch_times)

    # DaphneLib
    daphne_A = dctx.from_numpy(A, shared_memory=True)
    daphne_B = dctx.from_numpy(B, shared_memory=True)
    daphnelib_times = []
    for _ in range(repetitions):
        start = time.time()
        daphnelib_func(daphne_A, daphne_B)
        end = time.time()
        daphnelib_times.append(end - start)
    daphnelib_avg = np.mean(daphnelib_times)

    # Store results
    results["operation"].append(operation_name)
    results["size"].append(size)
    results["numpy"].append(numpy_avg)
    results["tensorflow"].append(tensorflow_avg)
    results["pandas"].append(pandas_avg)
    results["pytorch"].append(torch_avg)
    results["daphnelib"].append(daphnelib_avg)

# Define operations
def numpy_matmul(A, B):
    return A @ B

def tensorflow_matmul(A, B):
    return tf.matmul(A, B)

def pandas_matmul(A, B):
    return A.dot(B)

def pytorch_matmul(A, B):
    return torch.matmul(A, B)

def daphnelib_matmul(A, B):
    return (A @ B).compute()

def numpy_add(A, B):
    return A + B

def pandas_add(A, B):
    return A.add(B)

def tensorflow_add(A, B):
    return tf.add(A, B)

def pytorch_add(A, B):
    return torch.add(A, B)

def daphnelib_add(A, B):
    return (A + B).compute()

def benchmark_string_transfer(size):
    # Generate random string data
    strings = [f"string_{i}" for i in range(size)]

    # Convert to a 2D array (each string in its own row)
    strings_array = np.array(strings, dtype=object).reshape(-1, 1)

    # Transfer to DaphneLib
    start = time.time()
    daphne_strings = dctx.from_numpy(strings_array, shared_memory=False)
    transfer_to_daphne_time = time.time() - start

    # Store results
    results["operation"].append("String Transfer")
    results["size"].append(size)
    results["numpy"].append(0)  # Numpy does not support string transfer
    results["tensorflow"].append(0) # TensorFlow does not support string transfer
    results["daphnelib"].append(transfer_to_daphne_time)  # Using DaphneLib column for transfer to Daphne

    print(f"String transfer for size {size}:")
    print(f"  Transfer to DaphneLib: {transfer_to_daphne_time:.6f} seconds")

# Run benchmarks for different operations and matrix sizes
for size in [2000, 5000, 9000]:
    print(f"Running benchmarks for size: {size}x{size}")
    benchmark_operation("Matrix Multiplication", size, numpy_matmul, tensorflow_matmul, pandas_matmul, pytorch_matmul, daphnelib_matmul)
    benchmark_operation("Matrix Addition", size, numpy_add, tensorflow_add, pandas_add, pytorch_add, daphnelib_add)

# End-to-End Experiment: Linear Regression
def end_to_end_linear_regression(size):
    print(f"\nRunning end-to-end linear regression for size: {size}")
    # Generate random dataset
    X = np.random.rand(size, 10)
    y = np.random.rand(size)

    # Transfer to DaphneLib
    start = time.time()
    daphne_X = dctx.from_numpy(X, shared_memory=True)
    daphne_y = dctx.from_numpy(y, shared_memory=True)
    transfer_time = time.time() - start

    # Perform linear regression manually
    start = time.time()
    X_transpose = daphne_X.t()  # Transpose of X
    X_transpose_X = (X_transpose @ daphne_X).compute()  # X^T X
    X_transpose_y = (X_transpose @ daphne_y).compute()  # X^T y
    beta = np.linalg.solve(X_transpose_X, X_transpose_y)  # Solve for beta
    computation_time = time.time() - start

    # Transfer results back to Python
    start = time.time()
    result_numpy = beta  # Already in NumPy format
    result_transfer_time = time.time() - start

    print(f"Data transfer to DaphneLib: {transfer_time:.6f} seconds")
    print(f"Computation in DaphneLib: {computation_time:.6f} seconds")
    print(f"Result transfer to Python: {result_transfer_time:.6f} seconds")



# Run end-to-end experiment
end_to_end_linear_regression(1000)

# create bar plot for different operations
def create_bar_plot(data, title, file_path):
    plt.figure(figsize=(22, 5))
    plt.bar(np.arange(len(data["numpy"])), data["numpy"], label='Numpy', width=0.15, align='center')
    plt.bar(np.arange(len(data["tensorflow"])) + 0.15, data["tensorflow"], label='Tensorflow', width=0.15, align='center')
    plt.bar(np.arange(len(data["pandas"])) + 0.3, data["pandas"], label='Pandas', width=0.15, align='center')
    plt.bar(np.arange(len(data["pytorch"])) + 0.45, data["pytorch"], label='Pytorch', width=0.15, align='center')
    plt.bar(np.arange(len(data["daphnelib"])) + 0.6, data["daphnelib"], label='DaphneLib', width=0.15, align='center')
    plt.xlabel('Size of Matrices')
    plt.ylabel('seconds')
    plt.title(title)

    plt.xticks(np.arange(len(data["numpy"])) + 0.6 / 2, data["size"])
    plt.legend(loc='best')
    plt.savefig(file_path)

# Print results
print("\nBenchmark Results:")
print(f"{'Operation':<20}{'Size':<10}{'NumPy (s)':<15}{'TensorFlow (s)':<15}{'Pandas (s)':<15}{'Pytorch (s)':<15}{'DaphneLib (s)':<15}")
print("-" * 75)
# contains results of matrix multiplication benchmark
results_multi = {
    "size": [],
    "numpy": [],
    "tensorflow": [],
    "pandas": [],
    "pytorch": [],
    "daphnelib": []
}
# contains results of matrix addition benchmark
results_add = {
    "size": [],
    "numpy": [],
    "tensorflow": [],
    "pandas": [],
    "pytorch": [],
    "daphnelib": []
}
for i in range(len(results["operation"])):
    print(f"{results['operation'][i]:<20}{results['size'][i]:<10}{results['numpy'][i]:<15.6f}{results['tensorflow'][i]:<15.6f}{results['pandas'][i]:<15.6f}{results['pytorch'][i]:<15.6f}{results['daphnelib'][i]:<15.6f}")
    if "Matrix Multiplication" in results['operation'][i]:
        results_multi["size"].append(f"{results['size'][i]}x{results['size'][i]}")
        results_multi["numpy"].append(results['numpy'][i])
        results_multi["tensorflow"].append(results['tensorflow'][i])
        results_multi["pandas"].append(results['pandas'][i])
        results_multi["pytorch"].append(results['pytorch'][i])
        results_multi["daphnelib"].append(results['daphnelib'][i])
    elif "Matrix Addition" in results['operation'][i]:
        results_add["size"].append(f"{results['size'][i]}x{results['size'][i]}")
        results_add["numpy"].append(results['numpy'][i])
        results_add["tensorflow"].append(results['tensorflow'][i])
        results_add["pandas"].append(results['pandas'][i])
        results_add["pytorch"].append(results['pytorch'][i])
        results_add["daphnelib"].append(results['daphnelib'][i])

create_bar_plot(results_multi, "Matrix Multiplication", "benchmarks/api/python/matrix_multiplication_results.png")
create_bar_plot(results_add, "Matrix Addition", "benchmarks/api/python/matrix_addition_results.png")
