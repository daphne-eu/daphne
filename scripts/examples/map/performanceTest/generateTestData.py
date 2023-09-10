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

def measure_performance(command, max_timeout=600):
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
        result = ["bin/daphne", f"scripts/examples/map/performanceTest/performancetestScripts/daphneInternal/daphneInternal_{datatype}.daph", f"matrix_size={matrix_size}", f"minValue={min_value}", f"maxValue={max_value}", f"operation={operation}"]
    elif(implementation.startswith("Python_Ctypes")):
        result = ["bin/daphne", f"scripts/examples/map/performanceTest/performancetestScripts/mapExternal/mapExternalPLTest_{datatype}.daph", f"matrix_size={matrix_size}", f"minValue={min_value}", f"maxValue={max_value}", f"operation={operation}"]
        programming_language_arg = 'pl=\"' + implementation + '\"'
        result.append(programming_language_arg)
    elif(implementation == "Python_Numpy_Approach"):
        result = ["python3", f"scripts/examples/map/performanceTest/performancetestScripts/python_numpy_testscript.py",f"{datatype}" ,f"{operation}", f"{matrix_size}", f"{min_value}", f"{max_value}"]
    else:
        print(f"operation: {operation}, implementation: {implementation}, datatype: {datatype}, matrix_size: {matrix_size}, min_value: {min_value}, max_value: {max_value}")
    return result

def warmup_system_for_benchmarks(matrix_sizes, datatypes, implementations, operations):
    '''
    Warm-up logic to prepare the system for benchmarking.
    '''
    print("Warm Up System for Benchmarks")
    for _ in range(2):
        for impl in implementations:
            for dtype in datatypes:
                for op in operations:
                    if(op == 9 and dtype.startswith("f")): # Fibonacci function is not possible with floats
                        break
                    if((op == 8 or op == 4) and impl == "daphneInternal" and dtype.startswith("f")): # no cond and exp operations on float matrices
                        break
                    matrix_sizes_for_dtype = matrix_sizes.get(dtype)
                    for size in matrix_sizes_for_dtype:
                        min_for_op, max_for_op = getMinMaxValueRangeForOp(op)
                        command = generate_command(op, impl, dtype, size, min_for_op.get(dtype), max_for_op.get(dtype))
                        print(f"Warm-Up\n: {command}")
                        try:
                            subprocess.run(command, timeout=600)
                        except subprocess.TimeoutExpired:
                            print(f"Warning: Warm-up command '{command}' exceeded the timeout.")
    print("System Warm Up finish")

def run_benchmarks(matrix_sizes, datatypes, implementations, operations, runs=10, batch_size=0):
    
    def write_to_csv(batch_results):
        with open(f"scripts/examples/map/performanceTest/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(batch_results)

    warmup_system_for_benchmarks(matrix_sizes, datatypes, implementations, operations)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Open the file initially to write headers
    with open(f"scripts/examples/map/performanceTest/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Run", "Operation", "Implementation", "Datatype", "Matrix Size", "MetricType", "Value"])

    batch_results = []
    print("Start Test runs")
    timestamp_starting_benchmarks = time.time()
    for run in range(runs):
        for impl in implementations:
            for dtype in datatypes:
                for op in operations:
                    # Fibonacci function is not possible with floats
                    if(op == 9 and dtype.startswith("f")): 
                        break
                    # no conditional and exp operations on float matrices possible in daphne Internals
                    if((op == 8 or op == 4) and impl == "daphneInternal" and dtype.startswith("f")): 
                        break
                    matrix_sizes_for_dtype = matrix_sizes.get(dtype)
                    for size in matrix_sizes_for_dtype:
                        min_value_for_command, max_value_for_command = getMinMaxValueRangeForOp(op)
                        command = generate_command(op, impl, dtype, size, min_value_for_command.get(dtype), max_value_for_command.get(dtype))
                        print(f"Test run: {run} \nCommand:\n {command}")
                        try:
                            elapsed_time, max_memory, avg_cpu_load = measure_performance(command)
                            print(f"SUCCESS: elapsed_time: {elapsed_time}, max_memory: {max_memory}, avg_cpu_load: {avg_cpu_load}")
                            batch_results.extend([
                                (run, op, impl, dtype, size, "Execution Time", elapsed_time),
                                (run, op, impl, dtype, size, "Memory Consumption", max_memory),
                                (run, op, impl, dtype, size, "Average CPU Load", avg_cpu_load)
                            ])
                        except subprocess.TimeoutExpired:
                            print(f"Warning: Benchmark command '{command}' exceeded the timeout.")
                        
                        # Check if we reached the batch size, then write to CSV and clear the batch_results list
                        if len(batch_results) >= batch_size:
                            print("Write Results to csv")
                            write_to_csv(batch_results)
                            batch_results.clear()

    # After all loops, write any remaining results to the CSV
    if batch_results:
        print("Write Results to csv after Benchmark finish")
        write_to_csv(batch_results)
    timestamp_finish_benchmarks = time.time()
    end_time = timestamp_finish_benchmarks - timestamp_starting_benchmarks
    hours, minutes, secs = seconds_to_hms(end_time)
    print(f"Finished Benchmark in: {hours} hours, {minutes} minutes, {secs} seconds")

'Utility methods'

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return int(hours), int(minutes), int(seconds)

'Common min/max value range'
min_value = {
        'f64': -10,
        'f32': -10,
        'int64': -10,
        'int32': -10,
        'int8': -10,
        'uint64': 0,
        'uint8': 0
    }

max_value = {
        'f64': 10,
        'f32': 10,
        'int64': 10,
        'int32': 10,
        'int8': 10,
        'uint64': 10,
        'uint8': 10
    }

'Special Values for log'
min_value_for_log = {
        'f64': 1,
        'f32': 1,
        'int64': 1,
        'int32': 1,
        'int8': 1,
        'uint64': 1,
        'uint8': 1
}

'''For exponential function, we might need to further limit the input for int types to avoid quick overflows.'''
max_value_for_exp = {
        'f64': 5,
        'f32': 5,
        'int64': 5,
        'int32': 4,
        'int8': 2,
        'uint64': 5,
        'uint8': 2
}

min_value_for_exp = {
        'f64': -5,
        'f32': -5,
        'int64': -5,
        'int32': -4,
        'int8': -2,
        'uint64': 0,
        'uint8': 0
    }

'Polynomial'
min_value_polynomial = {
        'f64': -10,
        'f32': -10,
        'int64': -10,
        'int32': -10,
        'int8': -2,
        'uint64': 0,
        'uint8': 0
    }

max_value_polynomial = {
        'f64': 10,
        'f32': 10,
        'int64': 10,
        'int32': 10,
        'int8': 2,
        'uint64': 10,
        'uint8': 10
}

'Power'
min_value_power = {
        'f64': -10,
        'f32': -10,
        'int64': -10,
        'int32': -10,
        'int8': -5,
        'uint64': 0,
        'uint8': 0
    }

max_value_power = {
        'f64': 10,
        'f32': 10,
        'int64': 10,
        'int32': 10,
        'int8': 5,
        'uint64': 10,
        'uint8': 6
}

def getMinMaxValueRangeForOp(op):
    '''
    op1: multiplication
    op2: power
    op3: logarithm_base_10
    op4: e**(x**2)
    op5: polynomial
    op6: relu
    op7: sigmoid
    op8: thresholding
    op9: Fibonacci
    '''
    if op == 1:  # multiplication
        return min_value, max_value
    elif op == 2:  # power
        return min_value_power, max_value_power
    elif op == 3:  # logarithm_base_10
        return min_value_for_log, max_value
    elif op == 4:  # 2*e**(x**2)
        return min_value_for_exp, max_value_for_exp
    elif op == 5:  # polynomial
        return min_value_polynomial, max_value_polynomial
    elif op == 6:  # relu
        return min_value, max_value  # No change since relu outputs the input for positive values and 0 otherwise.
    elif op == 7:  # sigmoid
        return min_value, max_value  # No extreme restriction since output is bounded between 0 and 1.
    elif op == 8:  # thresholding
        return min_value, max_value
    elif op == 9: # fibonacci
        return min_value, max_value # No extreme restriction since it's a step function.
    else:
        raise ValueError("Invalid operation")

if __name__ == "__main__":

    matrix_sizes_1gb_2gb_3gb = {   
    'f32': [16384, 23000, 28377],
    'f64': [11585, 16000, 20066],
    'int32': [16384, 23000, 28377],
    'int64': [11585, 16000, 20066],
    'int8': [32768, 51000, 56755],
    'uint64': [11585, 16000, 20066],
    'uint8': [32768, 51000, 56755]
    }

    matrix_sizes_500mb_1gb = {   
    'f32': [8192, 16384],
    'f64': [5792, 11585],
    'int32': [8192, 16384],
    'int64': [5792, 11585],
    'int8': [16384, 32768],
    'uint64': [5792, 11585],
    'uint8': [16384, 32768]
    }

    matrix_sizes_250mb_500mb = {
    'f32': [4096, 8192],
    'f64': [2896, 5792],
    'int32': [4096, 8192],
    'int64': [2896, 5792],
    'int8': [8192, 16384],
    'uint64': [2896, 5792],
    'uint8': [8192, 16384]
    }

    matrix_sizes_3 = {
    'f32': [3],
    'f64': [3],
    'int32': [3],
    'int64': [3],
    'int8': [3],
    'uint64': [3],
    'uint8': [3]
    }

    datatypes = ['f64', 'f32', 'int64', 'int32', 'int8', 'uint64', 'uint8']
    datatypes_small_test = ['f64', 'f32', 'int64']

    implementations = ["daphneMap", "daphneInternal", "Python_Numpy_Approach", 
                       "Python_Ctypes_SysArg","Python_Ctypes_sharedMem_address", 
                       "Python_Ctypes_sharedMem_voidPointer", "Python_Ctypes_sharedMem_Pointer",
                        "Python_Ctypes_copy", "Python_Ctypes_binaryData", "Python_Ctypes_csv"]
    implementations_small_evaluation = ["daphneMap", "daphneInternal", "Python_Numpy_Approach", 
                       "Python_Ctypes_SysArg","Python_Ctypes_sharedMem_address", 
                        "Python_Ctypes_copy", "Python_Ctypes_binaryData", "Python_Ctypes_csv"]

    operations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    run_benchmarks(matrix_sizes_250mb_500mb, datatypes_small_test, implementations_small_evaluation, operations)