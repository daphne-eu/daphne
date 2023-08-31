import subprocess
import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

def measure_performance(command):
    '''
    Measure the execution time and memory consumpion for an executed command (here the execution of a DAPHNE script)
    '''
    start_time = time.time()
    mem_usage = memory_usage((subprocess.run, (command,)), interval=0.1, timeout=None, include_children=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    max_memory = max(mem_usage)
    return elapsed_time, max_memory

def run_experiment(matrix_sizes, function_complexities, matrix_type, test_type=None):
    '''
    function which runs the experiments of calling the DAPHNE map function and the DAPHNE/Python map function
    with the UDF x->x^(function_complexity) and Random matrix int the shape of matrix_sizes x matrix_sizes, 
    compares memory consumption and execution time of both functions and plots the results.
    '''
    results = {'time1': [], 'memory1': [], 'time2': [], 'memory2': []}
    total_diff_time = 0
    total_diff_memory = 0
    avg_time_diff = 0
    avg_memory_diff = 0

    min_diff_time = float('inf')
    max_diff_time = float('-inf')
    min_diff_memory = float('inf')
    max_diff_memory = float('-inf')
    #total_runs = 0  # To keep track of the total number of runs
    total_experiments = len(matrix_sizes) * len(function_complexities)

    for size in matrix_sizes:
        for complexity in function_complexities:
            #total_runs +=1

            map_function1_command = ["bin/daphne", f"scripts/examples/map/mapTest_{matrix_type}.daph", f"matrix_size={size}", f"minValue=1", f"maxValue=9", f"exponent={complexity}"]
            map_function2_command = ["bin/daphne", f"scripts/examples/map/mapExternalPLTest_{matrix_type}.daph", f"matrix_size={size}", f"minValue=1", f"maxValue=9", f"exponent={complexity}"]

            time1, memory1 = measure_performance(map_function1_command)
            time2, memory2 = measure_performance(map_function2_command)

            # total values
            diff_time = time2 - time1
            diff_memory = memory2 - memory1
            
            total_diff_time += diff_time
            total_diff_memory += diff_memory

            # set max and min values for memory consumption
            if(diff_memory > max_diff_memory):
                max_diff_memory = diff_memory
            if (diff_memory < min_diff_memory):
                min_diff_memory = diff_memory
            # set max and min values for execution time
            if(diff_time > max_diff_time):
                max_diff_time = diff_time
            if (diff_time < min_diff_time):
                min_diff_time = diff_time

            # percentage values
            time_diff_percent = ((time2 - time1) / time1) * 100
            memory_diff_percent = ((memory2 - memory1) / memory1) * 100
            
            avg_time_diff += time_diff_percent / total_experiments
            avg_memory_diff += memory_diff_percent / total_experiments
            
            results['time1'].append(time1)
            results['memory1'].append(memory1)
            results['time2'].append(time2)
            results['memory2'].append(memory2)

            print(  f"Matrix Size: {size}, Function Complexity: {complexity}\n" +
                    f"Time1: {time1:.4f} seconds, Memory1: {memory1:.4f} MiB\n" +
                    f"Time2: {time2:.4f} seconds, Memory2: {memory2:.4f} MiB")

    # Calculate the average difference - total
    avg_diff_time_total = total_diff_time / total_experiments
    avg_diff_memory_total = total_diff_memory / total_experiments

    print(f"Average Time Difference (total): {avg_diff_time_total:.4f} seconds")
    print(f"Average Memory Difference (total): {avg_diff_memory_total:.4f} MiB")

    # Calculate the average difference - percentage
    print(f"Average Time Deterioration (procentual difference): {avg_time_diff:.2f}%")
    print(f"Average Memory Deterioration (procentual difference): {avg_memory_diff:.2f}%")

    # Plotting
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(results['time1'])), results['time1'], label='Map Function DAPHNE - Execution Time')
    plt.plot(range(len(results['time2'])), results['time2'], label='Map Function Python - Execution Time')
    plt.xlabel('Experiment Index')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.annotate(f"Average Time Difference: {avg_diff_time_total:.4f} s", xy=(0, 1.13), xycoords='axes fraction')
    plt.annotate(f"Average Time Deterioration: {avg_time_diff:.4f} %", xy=(0, 1.09), xycoords='axes fraction')
    plt.annotate(f"Min Time Difference: {min_diff_time:.4f} s", xy=(0, 1.05), xycoords='axes fraction')
    plt.annotate(f"Max Time Difference: {max_diff_time:.4f} s", xy=(0, 1.01), xycoords='axes fraction')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(results['memory1'])), results['memory1'], label='Map Function DAPHNE - Memory Consumption')
    plt.plot(range(len(results['memory2'])), results['memory2'], label='Map Function Python - Memory Consumption')
    plt.xlabel('Experiment Index')
    plt.ylabel('Memory (MiB)')
    plt.legend()
    plt.annotate(f"Average Memory Difference: {avg_diff_memory_total:.4f} MiB", xy=(0, 1.13), xycoords='axes fraction')
    plt.annotate(f"Average Memory Deterioration: {avg_memory_diff:.4f} %", xy=(0, 1.09), xycoords='axes fraction')
    plt.annotate(f"Min Memory Difference: {min_diff_memory:.4f} MiB", xy=(0, 1.05), xycoords='axes fraction')
    plt.annotate(f"Max Memory Difference: {max_diff_memory:.4f} MiB", xy=(0, 1.01), xycoords='axes fraction')

    if(test_type != None):
        plt.savefig(f"scripts/examples/map/figures/MapPerformanceTest_{matrix_type}_{test_type}.png")
    else:
        plt.savefig(f"scripts/examples/map/figures/MapPerformanceTest_{matrix_type}.png")

def run_experiments_for_diff_types(matrix_sizes, function_complexities, matrix_types):
    '''
    The Experiments (determined by matrix size and function complexities) can be executed for different types.
    '''
    for type in matrix_types:
        run_experiment(matrix_sizes,function_complexities,type)

if __name__ == "__main__":
    matrix_sizes = [5, 10, 100]
    function_complexities = list(range(2, 18))
    run_experiment(matrix_sizes, function_complexities,"int64_t")