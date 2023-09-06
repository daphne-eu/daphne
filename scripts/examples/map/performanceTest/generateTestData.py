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

import subprocess
import time
from memory_profiler import memory_usage
import psutil
import csv
import datetime
import threading

def measure_performance(command, max_timeout=900): # 900 seconds = 15 minutes
    '''
    Measure the execution time, memory consumption, and CPU load for an executed command (the execution of a DAPHNE script)
    '''
    # To measure the initial CPU time
    start_cpu_time = psutil.cpu_times_percent(interval=None)
    start_time = time.time()
    
    # Capture CPU load periodically and store samples in cpu_loads list.
    cpu_loads = []

    def sample_cpu_load():
        while getattr(threading.current_thread(), "do_run", True):
            cpu_loads.append(psutil.cpu_percent(interval=1))
            time.sleep(0.5)
    
    t = threading.Thread(target=sample_cpu_load)
    t.start()
    
    try:
        mem_usage = memory_usage((subprocess.run, (command,)), interval=0.1, timeout=max_timeout, include_children=True) 
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' exceeded the {max_timeout//60} minutes timeout.")
        return "Timeout", "Timeout", "Timeout"
    finally:
        # Ensure to always stop the CPU sampling thread, regardless of whether the command timed out or completed
        t.do_run = False
        t.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    max_memory = max(mem_usage)
    avg_cpu_load = sum(cpu_loads)/len(cpu_loads) if cpu_loads else 0
    
    return elapsed_time, max_memory, avg_cpu_load

def generate_command(operation, implementation, datatype, matrix_size, min_value, max_value):
    '''
    Generates the command to invoke a script based on the implementation type
    '''
    if(implementation == "daphneMap"):
        result = ["bin/daphne", f"scripts/examples/map/performanceTest/performancetestScripts/daphneMap/mapTest_{datatype}.daph", f"matrix_size={matrix_size}", f"minValue={min_value}", f"maxValue={max_value}", f"operation={operation}"]
    elif(implementation == "daphneInternal"):
        result = ["bin/daphne", f"scripts/examples/map/performanceTest/performancetestScripts/daphneInternal/mapTest_{datatype}.daph", f"matrix_size={matrix_size}", f"minValue={min_value}", f"maxValue={max_value}", f"operation={operation}"]
    elif(implementation.startswith("Python_ctypes")):
        result = ["bin/daphne", f"scripts/examples/map/performanceTest/performancetestScripts/mapExternal/mapExternalPLTest_{datatype}.daph", f"matrix_size={matrix_size}", f"minValue={min_value}", f"maxValue={max_value}", f"operation={operation}"]
        programming_language_arg = 'pl=\"' + implementation + '\"'
        result.append(programming_language_arg)
    elif(implementation == "Python_Numpy_Approach"):
        result = ["python3", f"scripts/examples/map/performanceTest/performancetestScripts/python_numpy_testscript.py",f"{datatype}" ,f"{operation}", f"{matrix_size}", f"{min_value}", f"{max_value}"]
    else:
        print(f"operation: {operation}, implementation: {implementation}, datatype: {datatype}, matrix_size: {matrix_size}, min_value: {min_value}, max_value: {max_value}")
        raise RuntimeError("Wrong command")
    return result


def warmup_system_for_benchmarks(matrix_sizes, datatypes, implementations, operations):
    '''
    Warm-up logic to prepare the system for benchmarking.
    '''
    for _ in range(10):
        for size in matrix_sizes:
            for dtype in datatypes:
                for impl in implementations:
                    for op in operations:
                        min_for_op, max_for_op = getMinMaxValueRangeForOp(op)
                        command = generate_command(op, impl, dtype, size, min_for_op, max_for_op)
                        try:
                            subprocess.run(command, timeout=120)
                        except subprocess.TimeoutExpired:
                            print(f"Warning: Warm-up command '{command}' exceeded the timeout.")


def run_benchmarks_batch(matrix_sizes, datatypes, implementations, operations, runs=10, batch_size=1000):
    
    def write_to_csv(batch_results):
        with open(f"scripts/examples/map/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(batch_results)

    warmup_system_for_benchmarks(matrix_sizes, datatypes, implementations, operations)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Open the file initially to write headers
    with open(f"scripts/examples/map/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Operation", "Implementation", "Datatype", "Matrix Size", "MetricType", "Value"])

    batch_results = []

    for _ in range(runs):
        for size in matrix_sizes:
            for dtype in datatypes:
                for impl in implementations:
                    for op in operations:
                        min_value_for_command, max_value_for_command = getMinMaxValueRangeForOp(op)
                        command = generate_command(op, impl, dtype, size, min_value_for_command, max_value_for_command)
                        try:
                            elapsed_time, max_memory, avg_cpu_load = measure_performance(command)

                            batch_results.extend([
                                (op, impl, dtype, size, "Execution Time", elapsed_time),
                                (op, impl, dtype, size, "Memory Consumption", max_memory),
                                (op, impl, dtype, size, "Average CPU Load", avg_cpu_load)
                            ])
                        except subprocess.TimeoutExpired:
                            print(f"Warning: Benchmark command '{command}' exceeded the timeout.")

                        # Check if we reached the batch size, then write to CSV and clear the batch_results list
                        if len(batch_results) >= batch_size:
                            write_to_csv(batch_results)
                            batch_results.clear()

    # After all loops, write any remaining results to the CSV
    if batch_results:
        write_to_csv(batch_results)


def run_benchmarks_normal(matrix_sizes, datatypes, implementations, operations, runs=10):

    results = []
    warmup_system_for_benchmarks()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    for _ in range(runs):
        for size in matrix_sizes:
            for dtype in datatypes:
                for impl in implementations:
                    for op in operations:
                        min_value_for_command, max_value_for_command = getMinMaxValueRangeForOp(op)
                        command = generate_command(op, impl, dtype, size, min_value_for_command, max_value_for_command)
                        elapsed_time, max_memory, avg_cpu_load = measure_performance(command)

                        results.append((op, impl, dtype, size, "Execution Time", elapsed_time))
                        results.append((op, impl, dtype, size, "Memory Consumption", max_memory))
                        results.append((op, impl, dtype, size, "Average CPU Load", avg_cpu_load))

    with open(f"scripts/examples/map/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Operation", "Implementation", "Datatype", "Matrix Size", "MetricType", "Value"])
        writer.writerows(results)

'Common min/max value range for multiplication, trigonometrical, relu, hyperbolic tangent, thresholing, sigmoid'
min_value = {
        'double': -10.0,
        'float': -10.0,
        'int64_t': -10,
        'int32_t': -10,
        'int8_t': -10,
        'uint64_t': 0,
        'uint8_t': 0
    }

max_value = {
        'double': 10.0,
        'float': 10.0,
        'int64_t': 10,
        'int32_t': 10,
        'int8_t': 10,
        'uint64_t': 10,
        'uint8_t': 10
    }

'Special Values for log'
min_value_for_log = {
        'double': 1,
        'float': 1,
        'int64_t': 1,
        'int32_t': 1,
        'int8_t': 1,
        'uint64_t': 1,
        'uint8_t': 1
}

# For exponential function, we might need to further limit the input for int types to avoid quick overflows.
max_value_for_exp = {
        'double': 10.0,
        'float': 10.0,
        'int64_t': 5,
        'int32_t': 4,
        'int8_t': 2,
        'uint64_t': 5,
        'uint8_t': 2
}

'Polynomial'
min_value_polynomial = {
        'double': -10.0,
        'float': -10.0,
        'int64_t': -10,
        'int32_t': -10,
        'int8_t': -2,
        'uint64_t': 0,
        'uint8_t': 0
    }

max_value_polynomial = {
        'double': 10.0,
        'float': 10.0,
        'int64_t': 10,
        'int32_t': 10,
        'int8_t': 2,
        'uint64_t': 10,
        'uint8_t': 10
}

'Power'
min_value_power = {
        'double': -10.0,
        'float': -10.0,
        'int64_t': -10,
        'int32_t': -10,
        'int8_t': -5,
        'uint64_t': 0,
        'uint8_t': 0
    }

max_value_power = {
        'double': 10.0,
        'float': 10.0,
        'int64_t': 10,
        'int32_t': 10,
        'int8_t': 5,
        'uint64_t': 10,
        'uint8_t': 6
}

def getMinMaxValueRangeForOp(op):
    '''
    op1: multiplication
    op2: power
    op3: sinus
    op4: cosinus
    op5: logarithm
    op6: exponential
    op7: polynomial
    op8: relu
    op9: sigmoid
    op10: hyperbolic tangent
    op11: thresholding
    '''
    if((op == 1) or (op == 3) or (op == 4) or (op == 8) or (op == 9) or (op == 10) or (op == 11)):
        return min_value,max_value
    elif (op == 2):
        return min_value_power,max_value_power
    elif(op == 5):
        return min_value_for_log, max_value
    elif(op == 6):
        return min_value, max_value_for_exp
    elif( op == 7):
        return min_value_polynomial, max_value_polynomial

if __name__ == "__main__":

    datatypes = ['double', 'float', 'int64_t', 'int32_t', 'int8_t', 'uint64_t', 'uint8_t']
    matrix_sizes = {
        'double': 16000,   # 2 GB size
        'float': 23000,    # 2 GB size
        'int64_t': 16000,  # 2 GB size
        'int32_t': 23000,  # 2 GB size
        'int8_t': 51000,   # 2 GB size
        'uint64_t': 16000, # 2 GB size
        'uint8_t': 51000   # 2 GB size
    }
    operations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    implementations = []
    
    run_benchmarks_batch(matrix_sizes, datatypes, implementations, operations)