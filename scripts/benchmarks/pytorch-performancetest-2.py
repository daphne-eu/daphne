# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script benchmarks the performance of the daphne compute() fuction for pytorch tensors
# Execution Times for the execute() function execution, the PyTorch Tensor Transformation execution and overall execution are saved
# Results are saved to PyTorch-Compute-PerformaceTest.csv and Avg-PyTorch-Compute-PerformaceTest.csv

import torch

from api.python.context.daphne_context import DaphneContext

import pandas as pd
import numpy as np
from datetime import datetime
import io
import contextlib
import re

# Adjust based on the number of runs for this benchmark
runs = 50

# Creating a list of sizes for the objects
sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]

dc = DaphneContext()

text_stream = io.StringIO()

header = ['num_dim1', 'num_dim2', 'num_dim3', 'exec_func_exec_time', 'tensor_trans_time', 'overall_comp_exec_time', "tensor_size_mb"]
testData = []

# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0

# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0

print("\n\n###\n### PyTorch Tensor Compute Function Performance Test\n###\n")

# PYTORCH TENSORS 
# Creating a list of 3D TENSORS with different sizes
for idx, run in enumerate(range(runs)): 
    for size in sizes:
        # Create a 3DTensor with the given size
        tensor3d = torch.as_tensor(np.random.randn(size,size,size))
        #print(tensor3d)
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):
            print(f"Tensor length of Dim1:{list(tensor3d.shape)[0]}\n")
            print(f"Tensor length of Dim2:{list(tensor3d.shape)[1]}\n")
            print(f"Tensor length of Dim3:{list(tensor3d.shape)[2]}\n")

            # Transfer Tensor to DaphneLib
            F, orig_shape = dc.from_pytorch(tensor3d, return_shape=True)
            tensor3d = F.max(F).compute(isPytorch=True, verbose=True, shape=orig_shape)
        
        # Calculate the sizes for the tensor
        tensor_size_bytes = tensor3d.element_size() * tensor3d.nelement()
        tensor_size_mb = tensor_size_bytes / (1024 ** 2)
        total_size_gb += tensor_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Extract length of dim1 of tensor
        num_dim1_match = re.search(r'Tensor length of Dim1:(\d+)', captured_output)
        num_dim1 = int(num_dim1_match.group(1)) if num_dim1_match else None

        # Extract length of dim2 of tensor
        num_dim2_match = re.search(r'Tensor length of Dim2:(\d+)', captured_output)
        num_dim2 = int(num_dim2_match.group(1)) if num_dim2_match else None

        # Extract length of dim3 of tensor
        num_dim3_match = re.search(r'Tensor length of Dim3:(\d+)', captured_output)
        num_dim3 = int(num_dim3_match.group(1)) if num_dim3_match else None

        # Extract Execution Function Time
        exec_func_exec_time_match = re.search(r'Execute Function execution time:\s+([\d.]+) seconds', captured_output)
        exec_func_exec_time = float(exec_func_exec_time_match.group(1)) if exec_func_exec_time_match else None

        # Extract Computing Operation Time
        comp_op_time_match = re.search(r'PyTorch Tensor Transformation Execution time:\s+([\d.]+) seconds', captured_output)
        comp_op_time = float(comp_op_time_match.group(1)) if comp_op_time_match else None

        # Extract Overall Compute Time
        overall_comp_exec_time_match = re.search(r'Overall Compute Function execution time:\s+([\d.]+) seconds', captured_output)
        overall_comp_exec_time = float(overall_comp_exec_time_match.group(1)) if overall_comp_exec_time_match else None

        """
        # Debugging: Print the extracted values
        print(f'num_dim1: {num_dim1}')
        print(f'num_dim2: {num_dim2}')
        print(f'num_dim2: {num_dim2}')
        print(f'exec_func_exec_time: {exec_func_exec_time}')
        print(f'comp_op_time: {comp_op_time}')
        print(f'overall_comp_exec_time: {overall_comp_exec_time}')

        print(captured_output)
        """

        # Put the data together into a List
        data = [num_dim1, num_dim2, num_dim3, exec_func_exec_time, comp_op_time, overall_comp_exec_time, tensor_size_mb]
        testData.append(data)
        
        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 30
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        
        # Delete objects that are no longer need
        del tensor3d

        # Delete the Object in Daphne to prevent Memory Overflow
        F.delete()
        del F

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

print("Benchmark finished! [")
print()

# Create Frame for the Results
testResults = pd.DataFrame(testData, columns=header)

# Calculate average results
averageResults_list = []
for size in sizes:
    subset = testResults[testResults['num_dim1'] == size]
    average_row = subset.mean(axis=0).values.tolist()
    averageResults_list.append(average_row)

# Create a DataFrame from the intermediate array
averageResults = pd.DataFrame(averageResults_list, columns=header)

# Create a timestamp using the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

"""
print("\nFrame of all Results:")
print(testResults)
"""

print(f"\nAverage Results at {timestamp}:\n")
print(averageResults)  # This prints the frame of averages

testResults.to_csv(f'scripts/benchmarks/testoutputs/pytorch02_PyTorch-Compute-PerformaceTest_{timestamp}.csv', index=False)
averageResults.to_csv(f'scripts/benchmarks/testoutputs/pytorch02_Avg-PyTorch-Compute-PerformaceTest{timestamp}.csv', index=False)

print("\n###End of Performance Test.\n")
