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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def createBarplotsWithSeaborn(csv_file_name, operations=None, implementations=None, matrix_sizes=None, dtypes=None, selected_runs=None):
    data = pd.read_csv(csv_file_name)

    '''
    # Assuming you want to convert values for metrics related to memory. If the metric's name is different, adjust the condition accordingly.
    if 'Memory Consumption' in data['MetricType'].unique():
        # Filter rows with the 'Memory' metric and apply the conversion
        memory_rows = data['MetricType'] == 'Memory Consumption'
        print(f"Number of memory rows: {sum(memory_rows)}")
        data.loc[memory_rows, 'Value'] = data.loc[memory_rows, 'Value'].apply(convert_to_bytes)
    '''
    
    if operations is None:
        operations = data['Operation'].unique()
    if implementations is None:
        implementations = data['Implementation'].unique()
    if selected_runs:
        data = data[data['Run'].isin(selected_runs)]
    if matrix_sizes:
        data = data[data['Matrix Size'].isin(matrix_sizes)]
    if dtypes:
        data = data[data['Datatype'].isin(dtypes)]

    data = data[data['Operation'].isin(operations) & data['Implementation'].isin(implementations)]
    data['Operation'] = data['Operation'].apply(returnDescriptionToOp)

    metrics = data['MetricType'].unique()

    # Extract the timestamp from the CSV filename using regex
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    timestamp_match = re.search(timestamp_pattern, csv_file_name)
    timestamp = timestamp_match.group(1) if timestamp_match else "unknown_time"

    # Replace characters in the timestamp to make it filename-friendly
    clean_timestamp = timestamp.replace(":", "_").replace(" ", "_")

    # Create directory for the plots
    csv_file_basename = f"performance_results_{clean_timestamp}"
    plot_dir = os.path.join("scripts/examples/map/performanceTest/testdata/plots", csv_file_basename)
    os.makedirs(plot_dir, exist_ok=True)

    metric_limits = {}
    for metric in metrics:
        metric_data = data[data['MetricType'] == metric]
        metric_min = metric_data['Value'].min()
        metric_max = metric_data['Value'].max()
    
        # Adjusting the min and max values slightly for better visual representation
        metric_min -= 0.1 * metric_min
        metric_max += 0.1 * metric_max
    
        metric_limits[metric] = (metric_min, metric_max)


    for metric in metrics:
        subset_data = data[data['MetricType'] == metric]

        y_min_global, y_max_global = metric_limits[metric]

        unique_sizes = subset_data['Matrix Size'].unique()
        unique_dtypes = subset_data['Datatype'].unique()

        for size in unique_sizes:
            for dtype in unique_dtypes:
                plt.figure(figsize=(20, 12))
                specific_data = subset_data[(subset_data['Matrix Size'] == size) & (subset_data['Datatype'] == dtype)]

                # Calculate mean values for each Operation and Implementation combination
                means = specific_data.groupby(['Operation', 'Implementation'])['Value'].mean().reset_index()
                

                if not specific_data.empty:
                    y_min = specific_data['Value'].min()
                    y_max = specific_data['Value'].max()
                else:
                    print(f" For metric: {metric}, size: {size}, datatype: {dtype} the dataframe is empty")
                    plt.close()
                    continue  # Skip the plotting process for this iteration if the DataFrame is empty
                
                # Set y-axis limits for values
                '''
                if int(size) <= 1000:
                    if y_min == y_max:
                        y_min = y_min - 0.05 * y_min
                        y_max = y_max + 0.05 * y_max

                    y_min = means['Value'].min() - 0.1 * means['Value'].min()
                    y_max = means['Value'].max() + 0.1 * means['Value'].max()
                    plt.ylim([y_min, y_max])
                '''
                
                plt.ylim([y_min_global, y_max_global])

                ax = sns.barplot(data=means, x='Operation', y='Value', hue='Implementation', hue_order=implementations)

                # Display the values on top of the bars
                for p in ax.patches:
                    height = p.get_height()
                    if np.isnan(height):  # Skip NaN values
                        continue
                    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90)
                
                # Display names direct at the bar
                '''
                ax.legend_.remove()
                patches_per_hue = len(ax.patches) // len(implementations)
                for idx, patch in enumerate(ax.patches):
                    height = patch.get_height()
                    if np.isnan(height):  # Skip NaN values
                        continue
                    impl_name = implementations[idx // patches_per_hue]
                    ax.text(patch.get_x() + patch.get_width() / 2., height + (0.05 * (y_max_global - y_min_global)), impl_name, rotation=90, ha="center")
                '''

                if metric == "Memory Consumption":
                    plt.ylabel("Memory Consumption [MiB]")
                elif metric == "Execution Time":
                    plt.ylabel("Execution Time [s]")
                elif metric == "Average CPU Load":
                    plt.ylabel("Average CPU Load [%]")
                else:
                    plt.ylabel("Value")
                size_title = str(size)
                if(not size_title.endswith('mb')):
                   size_title = f"{size} rows x {size} cols"
                plt.title(f'Metric: {metric}, Matrix Size: {size_title}, Datatype: {dtype}')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                #plt.legend(loc='best')
                
                save_file_name = f"performance_results_{clean_timestamp}_bar_{metric}_{size}_{dtype}.png"
                save_path = os.path.join(plot_dir, save_file_name)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

def returnDescriptionToOp(op):
    operations = {
        1: 'x*2',
        2: 'x^3',
        3: 'Logarithm_base_10',
        4: 'e**(x**2)',
        5: 'Polynomial (5)',
        6: 'ReLu',
        7: 'Sigmoid',
        8: 'Thresholding',
        9: 'Fibonacci',
        10: 'Polynomial (3)'
    }
    return operations.get(op, op)  # if op not in dictionary, return op itself

def convert_to_bytes(value_in_mib):
    """
    Converts a value in MiB (Mebibytes) to bytes.
    """
    # There are 2**20 bytes in one Mebibyte (MiB)
    return int(value_in_mib * (2**20))

def modify_matrix_sizes_in_csv(csv_file_name):
    # Mapping logical sizes to physical sizes
    matrix_sizes_250mb = {
        'f32': [4096],
        'f64': [2896],
        'int32': [4096],
        'int64': [2896],
        'int8': [8192],
        'uint64': [2896],
        'uint8': [8192]
    }

    matrix_sizes_500mb = {
        'f32': [8192],
        'f64': [5792],
        'int32': [8192],
        'int64': [5792],
        'int8': [16384],
        'uint64': [5792],
        'uint8': [16384]
    }

    # Combining both dictionaries into one mapping
    combined_mapping = {}
    for dtype, sizes in matrix_sizes_250mb.items():
        for size in sizes:
            combined_mapping[(dtype, size)] = '250mb'

    for dtype, sizes in matrix_sizes_500mb.items():
        for size in sizes:
            combined_mapping[(dtype, size)] = '500mb'

    # Read the CSV into a DataFrame
    data = pd.read_csv(csv_file_name)

    # Replace old matrix sizes with new ones based on the mapping
    data['Matrix Size'] = data.apply(lambda row: combined_mapping.get((row['Datatype'], row['Matrix Size']), row['Matrix Size']), axis=1)

    # Construct the new filename
    base_name = os.path.basename(csv_file_name)
    name_without_extension, extension = os.path.splitext(base_name)
    new_filename = os.path.join(os.path.dirname(csv_file_name), f"{name_without_extension}_mapped_to_logical_datasizes{extension}")

    # Save the modified DataFrame to the new CSV file
    data.to_csv(new_filename, index=False)

    return new_filename  # Return the new filename for convenience

if __name__ == "__main__":
    'For local benchmark results'
    'Create Barpot 2: Comparison 2 Operations on all Implementations with one logic data size '
    createBarplotsWithSeaborn("scripts/examples/map/performanceTest/testdata/csv_files/local/performance_results_2023-09-17 02:12:17.csv",
                              selected_runs=[2,3,4,5,6,7,8,9], implementations=['daphneInternal', 'daphneMap', 'Python_Numpy_Approach', 'Python_Shared_Mem', 'Python_SysArg','Python_Copy', 'Python_BinaryFile'])

    'Create Barplot 1: Comparison 4 Operations on all Implementations with one physical datasize in f64'
    modify_matrix_sizes_in_csv("scripts/examples/map/performanceTest/testdata/csv_files/local/performance_results_2023-09-14 23:45:06.csv")
    createBarplotsWithSeaborn("scripts/examples/map/performanceTest/testdata/csv_files/local/performance_results_2023-09-14 23:45:06_mapped_to_logical_datasizes.csv", 
                              selected_runs=[2,3,4,5,6,7,8,9,10], implementations=['daphneInternal', 'daphneMap', 'Python_Numpy_Approach', 'Python_Shared_Mem', 'Python_SysArg','Python_Copy', 'Python_BinaryFile', 'Python_CsvFile'])

    'For VM Benchmark results'
    'Create Barpot 2: Comparison 2 Operations on all Implementations with one logic data size '
    createBarplotsWithSeaborn("scripts/examples/map/performanceTest/testdata/csv_files/vm1/performance_results_2023-09-16 16:17:29.csv",
                              selected_runs=[2,3,4,5,6,7,8,9], implementations=['daphneInternal', 'daphneMap', 'Python_Numpy_Approach', 'Python_Shared_Mem', 'Python_SysArg','Python_Copy', 'Python_BinaryFile', 'Python_CsvFile'])

    'Create Barplot 1: Comparison 4 Operations on all Implementations with one physical datasize in f64'
    modify_matrix_sizes_in_csv("scripts/examples/map/performanceTest/testdata/csv_files/vm1/performance_results_2023-09-16 20:53:11.csv")
    
    createBarplotsWithSeaborn("scripts/examples/map/performanceTest/testdata/csv_files/vm1/performance_results_2023-09-16 20:53:11_mapped_to_logical_datasizes.csv", 
                              selected_runs=[2,3,4,5,6,7,8,9,10], implementations=['daphneInternal', 'daphneMap', 'Python_Numpy_Approach', 'Python_Shared_Mem', 'Python_SysArg','Python_Copy', 'Python_BinaryFile', 'Python_CsvFile'])