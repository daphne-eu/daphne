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
import random
import datetime

def measure_performance(command):
    '''
    Measure the execution time, memory consumption, and CPU load for an executed command (the execution of a DAPHNE script)
    '''
    # To measure the initial CPU time
    start_cpu_time = psutil.cpu_times_percent(interval=None)
    start_time = time.time()
    # Capture CPU load periodically and store samples in cpu_loads list.
    cpu_loads = []
    def sample_cpu_load():
        while True:
            cpu_loads.append(psutil.cpu_percent(interval=1))
            time.sleep(0.5)
    
    import threading
    t = threading.Thread(target=sample_cpu_load)
    t.start()
    mem_usage = memory_usage((subprocess.run, (command,)), interval=0.1, timeout=None, include_children=True)
    # Stop the CPU sampling thread after the command completes
    t.do_run = False
    t.join()
    end_time = time.time()
    # To measure the final CPU time
    end_cpu_time = psutil.cpu_times_percent(interval=None)
    elapsed_time = end_time - start_time
    max_memory = max(mem_usage)
    # Compute average CPU load during the command execution
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
        RuntimeError("Wrong command")
    return result

def warmup_system_for_benchmarks():
    '''
    Warm-Up logic with specific values for the benchmarking with default values
    '''
    matrix_sizes = []
    datatypes = []
    implementations = []
    min_value = []
    max_value = []
    warmup_system(matrix_sizes, datatypes, implementations, min_value, max_value)

def warmup_system(matrix_sizes, datatypes, implementations, min_value, max_value):
    '''
    Warm-up logic to prepare the system for benchmarking.
    '''
    for _ in range(10):
        for size in matrix_sizes:
            for dtype in datatypes:
                for impl in implementations:
                    command = generate_command(
                        2,
                        random.choice(implementations),
                        random.choice(datatypes),
                        random.choice(matrix_sizes),
                        min_value,
                        max_value
                    )
                    try:
                        subprocess.run(command, timeout=120)
                    except subprocess.TimeoutExpired:
                        print(f"Warning: Warm-up command '{command}' exceeded the timeout.")

def run_benchmarks(matrix_sizes, datatypes, implementations, min_value, max_value, runs=10):

    results = []
    warmup_system_for_benchmarks()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    for _ in range(runs):
        for size in matrix_sizes:
            for dtype in datatypes:
                for impl in implementations:
                    command = generate_command("operation_name", impl, dtype, size, min_value, max_value)
                    elapsed_time, max_memory, avg_cpu_load = measure_performance(command)

                    results.append(("operation_name", impl, dtype, size, "Execution Time", elapsed_time))
                    results.append(("operation_name", impl, dtype, size, "Memory Consumption", max_memory))
                    results.append(("operation_name", impl, dtype, size, "Average CPU Load", avg_cpu_load))
        with open(f"scripts/examples/map/testdata/csv_files/performance_results_{formatted_datetime}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Operation", "Implementation", "Datatype", "Matrix Size", "MetricType", "Value"])
            writer.writerows(results)

if __name__ == "__main__":
    matrix_sizes = [100, 200, 300]
    datatypes = ['int', 'float']
    implementations = ['impl1', 'impl2']
    min_value = 0
    max_value = 1000
    
    run_benchmarks(matrix_sizes, datatypes, implementations, min_value, max_value)