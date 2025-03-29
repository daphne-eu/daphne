# In this file we compare the speed with which pandas, numpy, tensorflow, pytorch and plain python
# exchange data with daphne. We use different sizes of matrices in order to make a more accurate comparison and both file and shared memory transfers.
# Lastly, 3 graphs are created, that compare the different libraries and the two transfer types with one another.

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from daphne.context import DaphneContext
import tensorflow as tf
import torch

# Initialize Daphne context
dctx = DaphneContext()


results_exchange = {
    "shared_memory": [],
    "size": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_transfer = {
    "shared_memory": [],
    "size": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}
def benchmark_data_exchange(row_size, col_size, shared_memory=True, exchange=True):
    transfer_type= "shared memory" if shared_memory else "files"

    # Generate random data
    A = np.random.rand(row_size, col_size)
    B = np.random.rand(row_size, col_size)

    # data exchange with daphne via numpy
    startA = time.time()
    dx_np_A = dctx.from_numpy(A, shared_memory=shared_memory)
    if exchange:
        dx_np_A.compute(type=transfer_type)
    endA = time.time()
    time_A = endA - startA

    startB = time.time()
    dx_np_B = dctx.from_numpy(B, shared_memory=shared_memory)
    if exchange:
        dx_np_B.compute(type=transfer_type)
    endB = time.time()
    time_B = endB - startB

    np_avg = (time_A + time_B) / 2

    # data exchange with daphne via python lists
    pyA = A.tolist()
    pyB = B.tolist()
    startA = time.time()
    dx_py_A = dctx.from_python(pyA, shared_memory=shared_memory)
    if exchange:
        dx_py_A.compute(type=transfer_type)
    endA = time.time()
    time_A = endA - startA

    startB = time.time()
    dx_py_B = dctx.from_python(pyB, shared_memory=shared_memory)
    if exchange:
        dx_py_B.compute(type=transfer_type)
    endB = time.time()
    time_B = endB - startB

    py_avg = (time_A + time_B) / 2

    # data exchange with daphne via pandas
    pdA = pd.DataFrame(A)
    pdB = pd.DataFrame(B)
    startA = time.time()
    dx_pd_A = dctx.from_pandas(pdA, shared_memory=shared_memory)
    if exchange:
        dx_pd_A.compute(type=transfer_type)
    endA = time.time()
    time_A = endA - startA

    startB = time.time()
    dx_pd_B = dctx.from_pandas(pdB, shared_memory=shared_memory)
    if exchange:
        dx_pd_B.compute(type=transfer_type)
    endB = time.time()
    time_B = endB - startB

    pd_avg = (time_A + time_B) / 2

    # data exchange with daphne via tensorflow
    tf_A = tf.constant(A)
    tf_B = tf.constant(B)
    startA = time.time()
    dx_tf_A = dctx.from_tensorflow(tf_A, shared_memory=shared_memory)
    if exchange:
        dx_tf_A.compute(type=transfer_type)
    endA = time.time()
    time_A = endA - startA

    startB = time.time()
    dx_tf_B = dctx.from_tensorflow(tf_B, shared_memory=shared_memory)
    if exchange:
        dx_tf_B.compute(type=transfer_type)
    endB = time.time()
    time_B = endB - startB

    tf_avg = (time_A + time_B) / 2

    # data exchange with daphne via pytorch
    torch_A = torch.from_numpy(A)
    torch_B = torch.from_numpy(B)
    startA = time.time()
    dx_torch_A = dctx.from_pytorch(torch_A, shared_memory=shared_memory)
    if exchange:
        dx_torch_A.compute(type=transfer_type)
    endA = time.time()
    time_A = endA - startA

    startB = time.time()
    dx_torch_B = dctx.from_pytorch(torch_B, shared_memory=shared_memory)
    if exchange:
        dx_torch_B.compute(type=transfer_type)
    endB = time.time()
    time_B = endB - startB

    torch_avg = (time_A + time_B) / 2

    # store results
    if exchange:
        results_exchange["shared_memory"].append(shared_memory)
        results_exchange["size"].append([row_size, col_size])
        results_exchange["numpy"].append(np_avg)
        results_exchange["python"].append(py_avg)
        results_exchange["pandas"].append(pd_avg)
        results_exchange["pytorch"].append(torch_avg)
        results_exchange["tensorflow"].append(tf_avg)
    else:
        results_transfer["shared_memory"].append(shared_memory)
        results_transfer["size"].append([row_size, col_size])
        results_transfer["numpy"].append(np_avg)
        results_transfer["python"].append(py_avg)
        results_transfer["pandas"].append(pd_avg)
        results_transfer["pytorch"].append(torch_avg)
        results_transfer["tensorflow"].append(tf_avg)

    print(f"Running Benchmark for {transfer_type} transfer and matrix size of {row_size}x{col_size}.")


def create_bar_plot(data, title, file_path, x_axis_name, x_title):
    plt.figure(figsize=(20, 5))

    plt.bar(np.arange(len(data["numpy"])), data["numpy"], label='Numpy', width=0.15, align='center')
    plt.bar(np.arange(len(data["tensorflow"])) + 0.15, data["tensorflow"], label='Tensorflow', width=0.15, align='center')
    plt.bar(np.arange(len(data["pandas"])) + 0.3, data["pandas"], label='Pandas', width=0.15, align='center')
    plt.bar(np.arange(len(data["pytorch"])) + 0.45, data["pytorch"], label='Pytorch', width=0.15, align='center')
    plt.bar(np.arange(len(data["python"])) + 0.6, data["python"], label='Python', width=0.15, align='center')
    plt.xlabel(x_title)
    plt.ylabel('seconds')
    plt.title(title)

    plt.xticks(np.arange(len(data["numpy"])) + 0.6 / 2, data[x_axis_name])
    plt.legend(loc='best')
    plt.savefig(file_path)


# run benchmarks for different matrix sizes
for size in [100, 1000, 5000]:
    # run benchmark for nxn matrix for data transfer
    benchmark_data_exchange(size, size, exchange=False)
    benchmark_data_exchange(size, size, shared_memory=False, exchange=False)

    # run benchmark for nxn matrix for data exchange
    benchmark_data_exchange(size, size)
    benchmark_data_exchange(size, size, shared_memory=False)

# objects that store data for the different graphs that are created at the end of the file
results_exchange_5000 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_exchange_1000 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_exchange_100 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_transfer_5000 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_transfer_1000 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}

results_transfer_100 = {
    "shared_memory": [],
    "numpy": [],
    "pandas": [],
    "python": [],
    "pytorch": [],
    "tensorflow": []
}


# print results
print("\nData Transfer Results:")
print(f"{'Shared memory': <15}{'Size':<15}{'NumPy (s)':<15}{'TensorFlow (s)':<15}{'Pandas (s)':<15}{'Pytorch (s)':<15}{'Python (s)':<15}")
print("-" * 100)

for i in range(len(results_transfer["numpy"])):
    print(f"{results_transfer['shared_memory'][i]:<15}{results_transfer['size'][i][0]}x{results_transfer['size'][i][1]: <10}{results_transfer['numpy'][i]:<15.6f}{results_transfer['tensorflow'][i]:<15.6f}{results_transfer['pandas'][i]:<15.6f}{results_transfer['pytorch'][i]:<15.6f}{results_transfer['python'][i]:<15.6f}")

    if 5000 in results_transfer['size'][i]:
        results_transfer_5000["shared_memory"].append("Shared Memory" if results_transfer['shared_memory'][i] else "File Transfer")
        results_transfer_5000["numpy"].append(results_transfer['numpy'][i])
        results_transfer_5000["tensorflow"].append(results_transfer['tensorflow'][i])
        results_transfer_5000["pandas"].append(results_transfer['pandas'][i])
        results_transfer_5000["pytorch"].append(results_transfer['pytorch'][i])
        results_transfer_5000["python"].append(results_transfer['python'][i])

    if 1000 in results_transfer['size'][i]:
        results_transfer_1000["shared_memory"].append("Shared Memory" if results_transfer['shared_memory'][i] else "File Transfer")
        results_transfer_1000["numpy"].append(results_transfer['numpy'][i])
        results_transfer_1000["tensorflow"].append(results_transfer['tensorflow'][i])
        results_transfer_1000["pandas"].append(results_transfer['pandas'][i])
        results_transfer_1000["pytorch"].append(results_transfer['pytorch'][i])
        results_transfer_1000["python"].append(results_transfer['python'][i])

    if 100 in results_transfer['size'][i]:
        results_transfer_100["shared_memory"].append("Shared Memory" if results_transfer['shared_memory'][i] else "File Transfer")
        results_transfer_100["numpy"].append(results_transfer['numpy'][i])
        results_transfer_100["tensorflow"].append(results_transfer['tensorflow'][i])
        results_transfer_100["pandas"].append(results_transfer['pandas'][i])
        results_transfer_100["pytorch"].append(results_transfer['pytorch'][i])
        results_transfer_100["python"].append(results_transfer['python'][i])


# create graphs with results
create_bar_plot(results_transfer_5000, "Data Exchange of Matrix with size 5000x5000", "benchmarks/api/python/data_transfer_5000_matrix_results.png", "shared_memory", "Data Transfer Type")
create_bar_plot(results_transfer_1000, "Data Exchange of Matrix with size 1000x1000", "benchmarks/api/python/data_transfer_1000_matrix_results.png", "shared_memory", "Data Transfer Type")
create_bar_plot(results_transfer_100, "Data Exchange of Matrix with size 100x100", "benchmarks/api/python/data_transfer_100_matrix_results.png", "shared_memory", "Data Transfer Type")

##########################

print("\nData Exchange Results:")
print(f"{'Shared memory': <15}{'Size':<15}{'NumPy (s)':<15}{'TensorFlow (s)':<15}{'Pandas (s)':<15}{'Pytorch (s)':<15}{'Python (s)':<15}")
print("-" * 100)

for i in range(len(results_exchange["numpy"])):
    print(f"{results_exchange['shared_memory'][i]:<15}{results_exchange['size'][i][0]}x{results_exchange['size'][i][1]: <10}{results_exchange['numpy'][i]:<15.6f}{results_exchange['tensorflow'][i]:<15.6f}{results_exchange['pandas'][i]:<15.6f}{results_exchange['pytorch'][i]:<15.6f}{results_exchange['python'][i]:<15.6f}")

    if 5000 in results_exchange['size'][i]:
        results_exchange_5000["shared_memory"].append("Shared Memory" if results_exchange['shared_memory'][i] else "File Transfer")
        results_exchange_5000["numpy"].append(results_exchange['numpy'][i])
        results_exchange_5000["tensorflow"].append(results_exchange['tensorflow'][i])
        results_exchange_5000["pandas"].append(results_exchange['pandas'][i])
        results_exchange_5000["pytorch"].append(results_exchange['pytorch'][i])
        results_exchange_5000["python"].append(results_exchange['python'][i])

    if 1000 in results_exchange['size'][i]:
        results_exchange_1000["shared_memory"].append("Shared Memory" if results_exchange['shared_memory'][i] else "File Transfer")
        results_exchange_1000["numpy"].append(results_exchange['numpy'][i])
        results_exchange_1000["tensorflow"].append(results_exchange['tensorflow'][i])
        results_exchange_1000["pandas"].append(results_exchange['pandas'][i])
        results_exchange_1000["pytorch"].append(results_exchange['pytorch'][i])
        results_exchange_1000["python"].append(results_exchange['python'][i])

    if 100 in results_exchange['size'][i]:
        results_exchange_100["shared_memory"].append("Shared Memory" if results_exchange['shared_memory'][i] else "File Transfer")
        results_exchange_100["numpy"].append(results_exchange['numpy'][i])
        results_exchange_100["tensorflow"].append(results_exchange['tensorflow'][i])
        results_exchange_100["pandas"].append(results_exchange['pandas'][i])
        results_exchange_100["pytorch"].append(results_exchange['pytorch'][i])
        results_exchange_100["python"].append(results_exchange['python'][i])


# create graphs with results
create_bar_plot(results_exchange_5000, "Data Exchange of Matrix with size 5000x5000", "benchmarks/api/python/data_exchange_5000_matrix_results.png", "shared_memory", "Data Transfer Type")
create_bar_plot(results_exchange_1000, "Data Exchange of Matrix with size 1000x1000", "benchmarks/api/python/data_exchange_1000_matrix_results.png", "shared_memory", "Data Transfer Type")
create_bar_plot(results_exchange_100, "Data Exchange of Matrix with size 100x100", "benchmarks/api/python/data_exchange_100_matrix_results.png", "shared_memory", "Data Transfer Type")





