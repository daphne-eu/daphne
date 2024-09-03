import subprocess
import matplotlib.pyplot as plt
import re
import numpy as np
import json

def execute_daphne_script(script_path, additional_args=None, iterations=10):
    command = ["bin/daphne"]
    command.append("--select-matrix-repr")
    command.append("--statistics")
    if additional_args:
        command.extend(additional_args)
    command.append(script_path)
    
    sqrt_times = []
    add_times = []
    pattern = re.compile(r"\[info\]:\s+\d+\s+(_ewSqrt.*|_ewAdd.*)\s+(\d+\.\d+)")

    for _ in range(iterations):
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout
        for line in output.splitlines():            
            match = pattern.search(line)
            if match:
                time = float(match.group(2))
                operation = match.group(1)
                if "ewSqrt" in operation:
                    sqrt_times.append(time)
                elif "ewAdd" in operation:
                    add_times.append(time)
    
    avg_sqrt_time = sum(sqrt_times) / len(sqrt_times) if sqrt_times else 0
    avg_add_time = sum(add_times) / len(add_times) if add_times else 0
    
    return avg_sqrt_time, avg_add_time
  
def plot_results(results, save_path="stacked_performance_comparison_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    operations = ['Full Overlap', 'No Overlap', 'Random Overlap']
    num_operations = len(operations)
    bar_width = 0.35
    
    indices = np.arange(num_operations)
    
    inference_labels = ['Estimated Props.', 'Measured Props.']

    # Plotting
    for i, ((sqrt_no_args, add_no_args), (sqrt_with_args, add_with_args)) in enumerate(results):
        # Bottom position for the stacks
        bottom_no_args = [0, sqrt_no_args]  # Start at 0, stack add on top of sqrt
        bottom_with_args = [0, sqrt_with_args]

        # Estimated Properties (No Args)
        ax.bar(indices[i] - bar_width/2, [sqrt_no_args, add_no_args], bar_width, label='Sqrt No Args' if i == 0 else None, color='orange', bottom=bottom_no_args)
        ax.bar(indices[i] - bar_width/2, [add_no_args], bar_width, label='Add No Args' if i == 0 else None, color='blue', bottom=[sqrt_no_args])

        # Measured Properties (With Args)
        ax.bar(indices[i] + bar_width/2, [sqrt_with_args, add_with_args], bar_width, label='Sqrt With Args' if i == 0 else None, color='orange', bottom=bottom_with_args)
        ax.bar(indices[i] + bar_width/2, [add_with_args], bar_width, label='Add With Args' if i == 0 else None, color='blue', bottom=[sqrt_with_args])
        
        ax.text(indices[i] - bar_width/2, sqrt_no_args + add_no_args, inference_labels[0], ha='center', va='bottom', color='black', rotation=0)
        ax.text(indices[i] + bar_width/2, sqrt_with_args + add_with_args, inference_labels[1], ha='center', va='bottom', color='black', rotation=0)
        
    # Labels and aesthetics
    ax.set_xlabel('Operations')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Times for Sqrt and Add in Estimated Properties (left) and Propery Inference (right)')
    ax.set_xticks(indices)
    ax.set_xticklabels(operations)
    handles = [
        plt.Line2D([0], [0], color='blue', lw=4, label='Add'),
        plt.Line2D([0], [0], color='orange', lw=4, label='Sqrt')
    ]
    ax.legend(handles=handles, title="Operation Type", loc='lower left', bbox_to_anchor=(0, 0))
    
    plt.tight_layout()
    plt.savefig(save_path)

def perform_property_recording(script_path, additional_args=None):
    command = ["bin/daphne"]
    if additional_args:
        command.extend(additional_args)
    command.append("--enable_property_recording")
    command.append(script_path)
    
    subprocess.run(command, capture_output=True, text=True)

def save_results_to_json(results, filename="results.json"):
    data = {}
    operations = ['Full Overlap', 'No Overlap', 'Random Overlap']
    for op, result in zip(operations, results):
        data[op] = {
            "No Args": {"Sqrt": result[0][0], "Add": result[0][1]},
            "With Args": {"Sqrt": result[1][0], "Add": result[1][1]}
        }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def load_results_from_json(filepath="results.json"):
    # Convert JSON string to a Python dictionary
    with open(filepath, 'r') as file:
        data = json.load(file)
        
    # Extract data into the format expected by the plotting function
    results = []
    for operation, values in data.items():
        no_args = (values["No Args"]["Sqrt"], values["No Args"]["Add"])
        with_args = (values["With Args"]["Sqrt"], values["With Args"]["Add"])
        results.append((no_args, with_args))
        
    return results

def main():
    
    script_case1_path = "RuntimePropTests/RuntimePropTest_case1.daphne"
    script_case2_path = "RuntimePropTests/RuntimePropTest_case2.daphne"
    script_case3_path = "RuntimePropTests/RuntimePropTest_case3.daphne"
    iterations = 10

    print("Executing Operation 1")
    perform_property_recording(script_case1_path)
    op1_sqrt_no_args, op1_add_no_args = execute_daphne_script(script_case1_path, iterations=iterations)
    op1_sqrt_with_args, op1_add_with_args = execute_daphne_script(script_case1_path, additional_args=["--enable_property_insert"], iterations=iterations)
    print("Operation 1 finished")
    print("Executing Operation 2")
    perform_property_recording(script_case2_path)
    op2_sqrt_no_args, op2_add_no_args = execute_daphne_script(script_case2_path, iterations=iterations)
    op2_sqrt_with_args, op2_add_with_args = execute_daphne_script(script_case2_path, additional_args=["--enable_property_insert"], iterations=iterations)
    print("Operation 2 finished")
    print("Executing Operation 3")
    perform_property_recording(script_case3_path)
    op3_sqrt_no_args, op3_add_no_args = execute_daphne_script(script_case3_path, iterations=iterations)
    op3_sqrt_with_args, op3_add_with_args = execute_daphne_script(script_case3_path, additional_args=["--enable_property_insert"], iterations=iterations)
    print("Operation 3 finished")

    results = [
        ((op1_sqrt_no_args, op1_add_no_args), (op1_sqrt_with_args, op1_add_with_args)),
        ((op2_sqrt_no_args, op2_add_no_args), (op2_sqrt_with_args, op2_add_with_args)),
        ((op3_sqrt_no_args, op3_add_no_args), (op3_sqrt_with_args, op3_add_with_args))
    ]
    
    #results = load_results_from_json()
    save_results_to_json(results)
    plot_results(results)

if __name__ == "__main__":
    main()
