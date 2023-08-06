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

# This script demonstarted the capability of importing dataframes in various types from python into daphne with the from_pandas function
# Dataframes such as Series, Sparse Dataframe and Categorical Dataframe are converted to regular dataframes to be supported
# rbind() is performed as a computation 
# Results are printed in the console

from api.python.context.daphne_context import DaphneContext
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import time
import io
import contextlib
import re

dc = DaphneContext()

text_stream = io.StringIO()

#Create Lists and Dataframe for the PerfoamnceResults  
header = []
testData = []
dataframeTestData = []
seriesTestData = []
sparseDataframeTestData = []
categoricalDataframeTestData = []
testResults = pd.DataFrame()

# Adjust based on the number of runs for this benchmark
runs = 15

# Creating a list of sizes for the objects
sizes = [1, 10, 100, 1000, 10000, 100000]

# Initialize to prevent "cold start effects" for the Performance Test
for init_run in range(5):
    start_df = pd.DataFrame(np.random.randn(2, 2))
    dc.from_pandas(start_df)


print("\n\n###\n### Dataframes Performance Test\n###\n")
# DATAFRAME 
# Creating a list of dataframes with different sizes
dataframes = [pd.DataFrame(np.random.randn(size, 15)) for size in sizes]
# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0
# Looping through the dataframes and testing the from_pandas and compute operation
for idx, run in enumerate(range(runs)): 
    for df in dataframes: 
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):        
            # Transfer data to DaphneLib
            F = dc.from_pandas(df, verbose=True)
            
            # Appending 
            rbind_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            rbind_end_time = time.time()

            # Cartesian 
            cartesian_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            cartesian_end_time = time.time()

        # Calculate the size of the DataFrame in MB
        df_size_mb = df.memory_usage(index=True).sum() / (1024 ** 2)

        # Add the current size to the total size (convert to GB)
        total_size_gb += df_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Dataframe Type
        dataframe_type = "Dataframe"

        # shape
        dataframe_shape = df.shape

        # Extract type_check_exec_time
        type_check_exec_time_match = re.search(r'Frame Type Check Execution time:\s+([\d.]+) seconds', captured_output)
        type_check_exec_time = float(type_check_exec_time_match.group(1)) if type_check_exec_time_match else None

        # Extract rbind_time
        rbind_time = (rbind_end_time - rbind_start_time)

        # Extract cartesian_time
        cartesian_time = (cartesian_end_time - cartesian_start_time)

        # Extract all_cols_exec_time
        all_cols_exec_time_match = re.search(r'Execution time for all columns:\s+([\d.]+) seconds', captured_output)
        all_cols_exec_time = float(all_cols_exec_time_match.group(1)) if all_cols_exec_time_match else None

        # Extract overall_exec_time
        overall_exec_time_match = re.search(r'Overall Execution time:\s+([\d.]+) seconds', captured_output)
        overall_exec_time = float(overall_exec_time_match.group(1)) if overall_exec_time_match else None

        # Put the data together into a List
        data = [dataframe_type, dataframe_shape, type_check_exec_time, rbind_time, cartesian_time, all_cols_exec_time, overall_exec_time, df_size_mb]
        dataframeTestData.append(data)

        # Create headers in first run
        if(idx == 0):
            header = ['dataframe_type', 'shape', 'type_check_exec_time', 'rbind_time', 'cartesian_time', 'all_cols_exec_time', 'overall_exec_time', 'dataframe_size_mb']

        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 30
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

# Create Frame for the Results
DataframeTestResults = pd.DataFrame(dataframeTestData, columns=header)
testResults = pd.concat([testResults, DataframeTestResults])
#print(testResults)
print("\n###End of Dataframe Experiments.\n")


print("\n\n###\n### Series Performance Test\n###\n")
# SERIES
# Creating a list of series with different sizes
series = [pd.Series(np.random.randn(size)) for size in sizes]
# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0
# Looping through the series and testing the from_pandas and compute operation
for idx, run in enumerate(range(runs)): 
    for s in series:
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):  
            # Transfer data to DaphneLib
            F = dc.from_pandas(s, verbose=True)
                
            # Appending 
            rbind_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            rbind_end_time = time.time()

            # Cartesian 
            cartesian_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            cartesian_end_time = time.time()

        # Calculate the size of the series in MB
        df_size_mb = s.memory_usage(index=True) / (1024 ** 2)

        # Add the current size to the total size (convert to GB)
        total_size_gb += df_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Dataframe Type
        dataframe_type = "Series"

        # series_shape
        series_shape = s.shape

        # Extract type_check_exec_time
        type_check_exec_time_match = re.search(r'Frame Type Check Execution time:\s+([\d.]+) seconds', captured_output)
        type_check_exec_time = float(type_check_exec_time_match.group(1)) if type_check_exec_time_match else None

        # Extract rbind_time
        rbind_time = (rbind_end_time - rbind_start_time)

        # Extract cartesian_time
        cartesian_time = (cartesian_end_time - cartesian_start_time)

        # Extract all_cols_exec_time
        all_cols_exec_time_match = re.search(r'Execution time for all columns:\s+([\d.]+) seconds', captured_output)
        all_cols_exec_time = float(all_cols_exec_time_match.group(1)) if all_cols_exec_time_match else None

        # Extract overall_exec_time
        overall_exec_time_match = re.search(r'Overall Execution time:\s+([\d.]+) seconds', captured_output)
        overall_exec_time = float(overall_exec_time_match.group(1)) if overall_exec_time_match else None

        # Put the data together into a List
        data = [dataframe_type, series_shape, type_check_exec_time, rbind_time, cartesian_time, all_cols_exec_time, overall_exec_time, df_size_mb]
        seriesTestData.append(data)

        # Create headers in first run
        if(idx == 0):
            header = ['dataframe_type', 'shape', 'type_check_exec_time', 'rbind_time', 'cartesian_time', 'all_cols_exec_time', 'overall_exec_time', 'dataframe_size_mb']

        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 30
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

# Create Frame for the Results
SeriesTestResults = pd.DataFrame(seriesTestData, columns=header)
testResults = pd.concat([testResults, SeriesTestResults])
#print(testResults)
print("\n###End of Series Experiments.\n")


print("\n\n###\n### Sparse Dataframe Performance Test\n###\n")
# SPARSE DATAFRAME
# Creating a list of sparse dataframes with different sizes
sparse_dataframes = [pd.DataFrame({"A": pd.arrays.SparseArray(np.random.randn(size)), "B": pd.arrays.SparseArray(np.random.randn(size)), "C": pd.arrays.SparseArray(np.random.randn(size))}) for size in sizes]
# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0
# Looping through the sparse dataframe and testing the from_pandas and compute operation
for idx, run in enumerate(range(runs)): 
    for sdf in sparse_dataframes:
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):  
            # Transfer data to DaphneLib
            F = dc.from_pandas(sdf, verbose=True)
                
            # Appending 
            rbind_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            rbind_end_time = time.time()

            # Cartesian 
            cartesian_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            cartesian_end_time = time.time()

        # Calculate the size of the series in MB
        df_size_mb = sdf.memory_usage(index=True).sum() / (1024 ** 2)

        # Add the current size to the total size (convert to GB)
        total_size_gb += df_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Dataframe Type
        dataframe_type = "Sparse Dataframe"

        # sparse_dataframe_shape
        sparse_dataframe_shape = sdf.shape

        # Extract type_check_exec_time
        type_check_exec_time_match = re.search(r'Frame Type Check Execution time:\s+([\d.]+) seconds', captured_output)
        type_check_exec_time = float(type_check_exec_time_match.group(1)) if type_check_exec_time_match else None

        # Extract rbind_time
        rbind_time = (rbind_end_time - rbind_start_time)

        # Extract cartesian_time
        cartesian_time = (cartesian_end_time - cartesian_start_time)

        # Extract all_cols_exec_time
        all_cols_exec_time_match = re.search(r'Execution time for all columns:\s+([\d.]+) seconds', captured_output)
        all_cols_exec_time = float(all_cols_exec_time_match.group(1)) if all_cols_exec_time_match else None

        # Extract overall_exec_time
        overall_exec_time_match = re.search(r'Overall Execution time:\s+([\d.]+) seconds', captured_output)
        overall_exec_time = float(overall_exec_time_match.group(1)) if overall_exec_time_match else None

        # Put the data together into a List
        data = [dataframe_type, sparse_dataframe_shape, type_check_exec_time, rbind_time, cartesian_time, all_cols_exec_time, overall_exec_time, df_size_mb]
        sparseDataframeTestData.append(data)

        # Create headers in first run
        if(idx == 0):
            header = ['dataframe_type', 'shape', 'type_check_exec_time', 'rbind_time', 'cartesian_time', 'all_cols_exec_time', 'overall_exec_time', 'dataframe_size_mb']

        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 30
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

# Create Frame for the Results
SparseDataframeTestResults = pd.DataFrame(sparseDataframeTestData, columns=header)
testResults = pd.concat([testResults, SparseDataframeTestResults])
#print(testResults)
print("\n###End of Sparse DataFrame Experiments.\n")


print("\n\n###\n### Categorical Dataframe Performance Test\n###\n")
# CATEGORICAL DATAFRAME
# Creating a list of categorical dataframes with different sizes
categorical_dataframes = [pd.DataFrame(np.random.randn(size, 15)) for size in sizes]
# Set the iterations for the progress bar
total_iterations = len(sizes) * runs
current_iteration = 0
total_size_gb = 0.0
# Looping through the categorical dataframe and testing the from_pandas and compute operation
for idx, run in enumerate(range(runs)): 
    for cdf in categorical_dataframes:
        cdf = cdf.astype("category")
        # Capture Verbose Outputs
        with contextlib.redirect_stdout(text_stream):  
            # Transfer data to DaphneLib
            F = dc.from_pandas(cdf, verbose=True)
                
            # Appending 
            rbind_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            rbind_end_time = time.time()

            # Cartesian 
            cartesian_start_time = time.time()
            F = F.rbind(F)
            F.compute()
            cartesian_end_time = time.time()

        # Calculate the size of the series in MB
        df_size_mb = cdf.memory_usage(index=True).sum() / (1024 ** 2)

        # Add the current size to the total size (convert to GB)
        total_size_gb += df_size_mb / 1024

        # Reset to the beginning of the text stream
        text_stream.seek(0)

        # Read the content of the text stream
        captured_output = text_stream.read()

        # Dataframe Type
        dataframe_type = "Categorical Dataframe"

        # categorical_dataframe_shape
        categorical_dataframe_shape = cdf.shape

        # Extract type_check_exec_time
        type_check_exec_time_match = re.search(r'Frame Type Check Execution time:\s+([\d.]+) seconds', captured_output)
        type_check_exec_time = float(type_check_exec_time_match.group(1)) if type_check_exec_time_match else None

        # Extract rbind_time
        rbind_time = (rbind_end_time - rbind_start_time)

        # Extract cartesian_time
        cartesian_time = (cartesian_end_time - cartesian_start_time)

        # Extract all_cols_exec_time
        all_cols_exec_time_match = re.search(r'Execution time for all columns:\s+([\d.]+) seconds', captured_output)
        all_cols_exec_time = float(all_cols_exec_time_match.group(1)) if all_cols_exec_time_match else None

        # Extract overall_exec_time
        overall_exec_time_match = re.search(r'Overall Execution time:\s+([\d.]+) seconds', captured_output)
        overall_exec_time = float(overall_exec_time_match.group(1)) if overall_exec_time_match else None

        # Put the data together into a List
        data = [dataframe_type, categorical_dataframe_shape, type_check_exec_time, rbind_time, cartesian_time, all_cols_exec_time, overall_exec_time, df_size_mb]
        categoricalDataframeTestData.append(data)

        # Create headers in first run
        if(idx == 0):
            header = ['dataframe_type', 'shape', 'type_check_exec_time', 'rbind_time', 'cartesian_time', 'all_cols_exec_time', 'overall_exec_time', 'dataframe_size_mb']

        # Clear the text stream for the next iteration
        text_stream.seek(0)
        text_stream.truncate(0)

        # Update the progress bar
        current_iteration += 1
        progress_percentage = (current_iteration / total_iterations) * 100
        bar_length = 30
        completed_length = int(bar_length * current_iteration // total_iterations)
        progress_bar = "#" * completed_length + "-" * (bar_length - completed_length)

        print(f'Progress: [{progress_bar}] {progress_percentage:.2f}% - Run {run + 1} - Total Size Processed: {total_size_gb:.3f} GB', end="\r")

# Create Frame for the Results
CategoricalDataframeTestResults = pd.DataFrame(categoricalDataframeTestData, columns=header)
testResults = pd.concat([testResults, CategoricalDataframeTestResults])
print("\n###End of Categorical DataFrame Experiments.\n")
#print(testResults)

"""
# MULTIINDEX DATAFRAME - Since A Conversion is not possible yet, the script would fail here..
# Creating a list of multiindex dataframes with different sizes
multiindex_dataframes = [pd.MultiIndex.from_product([list('"AB"'), range(size)]) for size in sizes]
# Looping through the multiindex dataframes and testing the from_pandas and compute operation
for midf in multiindex_dataframes:
    # Transfer data to DaphneLib
    F = dc.from_pandas(midf)
    # Appending 
    F = F.rbind(F)
    F.compute()
    print(F.compute())
print("\n###End of MultiIndex Experiments.\n")
"""

# Create a timestamp using the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")


#print(f"\nAverage Results at {timestamp}:\n")
#print(averageResults)  # This prints the frame of averages

testResults.to_csv(f'scripts/benchmarks/testoutputs/pandas01_PandasDataframes-PerformaceTest_{timestamp}.csv', index=False)
#averageResults.to_csv(f'scripts/benchmarks/testoutputs/{timestamp}_Avg-FromPandas-PerformaceTest.csv', index=False)
