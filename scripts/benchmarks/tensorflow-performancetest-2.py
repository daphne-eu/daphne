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

# This script will test the performance of the daphne compute fuction for tensorflow tensors

from api.python.context.daphne_context import DaphneContext

import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import io
import contextlib
import re

# Adjust based on the number of runs for this benchmark
runs = 15

# Creating a list of sizes for the objects
sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]

dc = DaphneContext()

text_stream = io.StringIO()

header = ['num_dim1', 'num_dim2', 'num_dim3', 'exec_func_exec_time', 'tensor_trans_exec_time', 'overall_comp_exec_time']
testData = []

# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0

# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0

print("\n\n###\n### Tensorflow Tensor Compute Function Performance Test\n###\n")

# PYTORCH TENSORS 
# Creating a list of 3D TENSORS with different sizes
for idx, run in enumerate(range(runs)): 
    for size in sizes:
        # Create a 3DTensor with the given size
        tensor3d = tf.constant(np.random.randn(size,size,size))
        #print(tensor3d)
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):
            print(f"Tensor length of Dim1:{tensor3d.get_shape()[0]}\n")
            print(f"Tensor length of Dim2:{tensor3d.get_shape()[1]}\n")
            print(f"Tensor length of Dim3:{tensor3d.get_shape()[2]}\n")

            # Transfer Tensor to DaphneLib
            F = dc.from_tensorflow(tensor3d)
            tensor3d = F.max(F).compute(isTensorflow=True, verbose=True)

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

        # Extract Tensor Transformation Execution Time
        tensor_trans_exec_time_match = re.search(r'TensorFlow Tensor Transformation Execution time:\s+([\d.]+) seconds', captured_output)
        tensor_trans_exec_time = float(tensor_trans_exec_time_match.group(1)) if tensor_trans_exec_time_match else None

        # Extract Overall Compute Time
        overall_comp_exec_time_match = re.search(r'Overall Compute Function execution time:\s+([\d.]+) seconds', captured_output)
        overall_comp_exec_time = float(overall_comp_exec_time_match.group(1)) if overall_comp_exec_time_match else None

        
        # Debugging: Print the extracted values
        print(f'num_dim1: {num_dim1}')
        print(f'num_dim2: {num_dim2}')
        print(f'num_dim2: {num_dim2}')
        print(f'exec_func_exec_time: {exec_func_exec_time}')
        print(f'comp_op_time: {tensor_trans_exec_time}')
        print(f'overall_comp_exec_time: {overall_comp_exec_time}')

        print(captured_output)
        

        # Put the data together into a List
        data = [num_dim1, num_dim2, num_dim3, exec_func_exec_time, tensor_trans_exec_time, overall_comp_exec_time]
        testData.append(data)
        
        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 40
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        
        # Delete objects that are no longer need
        del tensor3d

        # Delete the Object in Daphne to prevent Memory Overflow
        F.delete()
        del F

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1}', end="\r")

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

testResults.to_csv(f'scripts/benchmarks/testoutputs/{timestamp}TensorFlow-Compute-PerformaceTest.csv', index=False)
averageResults.to_csv(f'scripts/benchmarks/testoutputs/{timestamp}_Avg-TensorFlow-Compute-PerformaceTest.csv', index=False)

print("\n###End of Performance Test.\n")
