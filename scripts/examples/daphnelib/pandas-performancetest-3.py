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

from api.python.context.daphne_context import DaphneContext
import pandas as pd
import numpy as np
import csv
import time
import io
import contextlib
import re


# Creating a list of sizes for the objects
sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]


dc = DaphneContext()

text_stream = io.StringIO()

header = []
testData = []

# DATAFRAME 
# Creating a list of dataframes with different sizes
dataframes = [pd.DataFrame(np.random.randn(size, 15)) for size in sizes]
# Looping through the dataframes and testing the from_pandas and compute operation
print("\n\n###\n### Dataframe Performance Experiments:\n###\n")
for idx, df in enumerate(dataframes):

    # Capture Verbose Outputs
    with contextlib.redirect_stdout(text_stream):
        print(f"Test with num rows:\n{df.shape[0]}\n")
        print(f"Test with num cols:\n{df.shape[1]}\n")
        # Transfer data to DaphneLib
        F = dc.from_pandas(df)
        F = F.rbind(F).compute(verbose=True)
        #print(F.compute(verbose=True))
        #F.print().compute(verbose=True)

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
    data = [num_rows, num_cols, exec_func_exec_time, comp_op_time, overall_comp_exec_time]
    testData.append(data)

    # Create headers in first run
    if(idx == 0):
        header = ['num_rows', 'num_cols', 'exec_func_exec_time', 'comp_op_time', 'overall_comp_exec_time']
    
    # Clear the text stream for the next iteration
    text_stream.seek(0)
    text_stream.truncate(0)

# Create Frame for the Results
testResults = pd.DataFrame(testData, columns=header)

print("\nFrame from Output:")
print(testResults)

testResults.to_csv('scripts/examples/daphnelib/testoutputs/Pandas-Compute-PerformaceTest.csv', index=False)


print("\n###End of Performance Experiments.\n")
