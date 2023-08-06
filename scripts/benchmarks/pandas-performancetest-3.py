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

# This script benchmarks the performance of the daphne compute() function for pandas dataframes
# Execution Times for the execute() function, the computing operation of building the dataframe and overall compute() function are saved
# Results are saved to Pandas-Compute-PerformaceTest.csv and Avg-Pandas-Compute-PerformaceTest.csv

from api.python.context.daphne_context import DaphneContext
import pandas as pd
import numpy as np
from datetime import datetime
import io
import contextlib
import re

# Adjust based on the number of runs for this benchmark
runs = 10

# Creating a list of sizes for the objects
sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]#, 10000000, 50000000]
cols = 25

# Different Data Types for the benchmark
dtypes_list = [np.double, np.float64, np.float32, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

dc = DaphneContext()

text_stream = io.StringIO()

header = ['num_rows', 'num_cols', 'exec_func_exec_time', 'comp_op_time', 'overall_comp_exec_time', 'dataframe_size_mb']
testData = []

# Initialize to prevent "cold start effects" for the Performance Test
for init_run in range(5):
    start_df = pd.DataFrame(np.random.randn(2, 2))
    start_F = dc.from_pandas(start_df)
    start_F.rbind(start_F).compute()

# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0

print("\n\n###\n### Dataframe Compute Function Performance Test\n###\n")

# DATAFRAME 
# Creating a list of dataframes with different sizes
for run in range(runs): 
    for size in sizes:
        df_data = {}  # Initialize df_data as a dictionary
        
        # Create Arrays with random Data Types for each column
        for col in range(cols): 
            dtype = np.random.choice(dtypes_list)
            if np.issubdtype(dtype, np.floating):
                df_data[col] = np.random.randn(size).astype(dtype)
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger):
                high = np.iinfo(dtype).max
                if dtype == np.uint64 or dtype == np.int64:
                    high = np.iinfo(np.int64).max - 1  # Limit to the maximum value of uint64 to the value for int64
                else:
                    high -= 1
                low = np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else 0
                df_data[col] = np.random.randint(low, high, size=size).astype(dtype)

        # Create a Data Frame with the data from the arrays above
        df = pd.DataFrame(df_data, copy=False)

        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):
            print(f"Test with num rows:\n{df.shape[0]}\n")
            print(f"Test with num cols:\n{df.shape[1]}\n")
            # Transfer data to DaphneLib
            F = dc.from_pandas(df)
            df = F.rbind(F).compute(verbose=True)

        # Calculate the size of the DataFrame in MB
        df_size_mb = df.memory_usage(index=True).sum() / (1024 ** 2)
        
        # Add the current size to the total size (convert to GB)
        total_size_gb += df_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Extract num_rows
        num_rows_match = re.search(r'Test with num rows:\n(\d+)', captured_output)
        num_rows = int(num_rows_match.group(1)) if num_rows_match else None

        # Extract num_cols
        num_cols_match = re.search(r'Test with num cols:\n(\d+)', captured_output)
        num_cols = int(num_cols_match.group(1)) if num_cols_match else None

        # Extract Execution Function Time
        exec_func_exec_time_match = re.search(r'Execute Function execution time:\s+([\d.]+) seconds', captured_output)
        exec_func_exec_time = float(exec_func_exec_time_match.group(1)) if exec_func_exec_time_match else None

        # Extract Computing Operation Time
        comp_op_time_match = re.search(r'Computing Operation execution time:\s+([\d.]+) seconds', captured_output)
        comp_op_time = float(comp_op_time_match.group(1)) if comp_op_time_match else None

        # Extract Overall Compute Time
        overall_comp_exec_time_match = re.search(r'Overall Compute Function execution time:\s+([\d.]+) seconds', captured_output)
        overall_comp_exec_time = float(overall_comp_exec_time_match.group(1)) if overall_comp_exec_time_match else None

        """
        # Debugging: Print the extracted values
        print(f'num_rows: {num_rows}')
        print(f'num_cols: {num_cols}')
        print(f'exec_func_exec_time: {exec_func_exec_time}')
        print(f'comp_op_time: {comp_op_time}')
        print(f'overall_comp_exec_time: {overall_comp_exec_time}')

        print(captured_output)
        """

        # Put the data together into a List
        data = [num_rows, num_cols, exec_func_exec_time, comp_op_time, overall_comp_exec_time, df_size_mb]
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
        #del df

        # Delete the Object in Daphne to prevent Memory Overflow
        F.delete()
        #del F

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

print("Benchmark finished! [")
print()

# Create Frame for the Results
testResults = pd.DataFrame(testData, columns=header)

# Calculate average results
averageResults_list = []
for size in sizes:
    subset = testResults[testResults['num_rows'] == size]
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

testResults.to_csv(f'scripts/benchmarks/testoutputs/pandas03_Pandas-Compute-PerformaceTest_{timestamp}.csv', index=False)
averageResults.to_csv(f'scripts/benchmarks/testoutputs/pandas03_Avg-Pandas-Compute-PerformaceTest_{timestamp}.csv', index=False)

print("\n###End of Performance Test.\n")
