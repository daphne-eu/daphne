import subprocess
import matplotlib.pyplot as plt
import re
import numpy as np
import json
import matplotlib.ticker as ticker


def extract_matrix_type(operation):
    match = re.search(r"(DenseMatrix|CSRMatrix)_double", operation)
    return match.group(1) if match else "Unknown"

def truncate_to_four_decimals(value):
    try:
        formatted_value = f"{float(value):.4f}"
        return float(formatted_value)
    except ValueError:
        return value

def execute_daphne_script(script_path, additional_args=None, iterations=1, estimate_res_sparsity=True):
    command = ["bin/daphne"]
    command.append("--select-matrix-repr")
    command.append("--statistics")
    if additional_args:
        command.extend(additional_args)
    command.append(script_path)

    RC_matmul_times = []
    CR_matmul_times = []
    Avg_matmul_times = []
    RC_mamtmul_operations = []
    CR_mamtmul_operations = []
    Avg_mamtmul_operations = []

    pattern = re.compile(r"\[info\]:\s+(\d+)\s+(_matMul.*)\s+(\d+\.\d+)\s+.*:(\d+):\d+")

    RC_line_number = 7
    CR_line_number = 14
    Avg_line_number = 21
    
    for _ in range(iterations):
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout
        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                operation_idx = int(match.group(1))
                operation = match.group(2)
                time = float(match.group(3))
                line_number = int(match.group(4))  # Get the line number from the file

                # Classify based on line number (RC ,CR or Avg)
                if line_number == RC_line_number:
                    RC_matmul_times.append(time)
                    RC_mamtmul_operations.append(extract_matrix_type(operation))
                elif line_number == CR_line_number:
                    CR_matmul_times.append(time)
                    CR_mamtmul_operations.append(extract_matrix_type(operation))
                elif line_number == Avg_line_number:
                    Avg_matmul_times.append(time)
                    Avg_mamtmul_operations.append(extract_matrix_type(operation))
                else:
                    print(f"Unclassified line number: {line_number}")  # For unexpected cases

    RC_avg_matmul_time = sum(RC_matmul_times) / len(RC_matmul_times) if RC_matmul_times else 0
    CR_avg_matmul_time = sum(CR_matmul_times) / len(CR_matmul_times) if CR_matmul_times else 0
    Avg_avg_matmul_time = sum(Avg_matmul_times) / len(Avg_matmul_times) if Avg_matmul_times else 0
    
    RC_mamtmul_operation = RC_mamtmul_operations[-1] if RC_mamtmul_operations else "Unknown"
    CR_mamtmul_operation = CR_mamtmul_operations[-1] if CR_mamtmul_operations else "Unknown"
    Avg_mamtmul_operation = Avg_mamtmul_operations[-1] if Avg_mamtmul_operations else "Unknown"

    if estimate_res_sparsity:
        #result = subprocess.run(command, capture_output=True, text=True)
        #output = result.stdout
        sparsity_pattern = re.compile(r"sparsity\s(RC|CR|Avg):\s*(\d+\.\d+)")
        res_sparsity = {"RC": 0.0, "CR": 0.0, "Avg": 0.0}
        for line in output.splitlines():
            match = sparsity_pattern.search(line)
            if match:
                matrix_type = match.group(1)  # either RC, CR or Avg
                sparsity_value = float(match.group(2))
                res_sparsity[matrix_type] = truncate_to_four_decimals(sparsity_value)

        RC_res_sparsity = res_sparsity["RC"]
        CR_res_sparsity = res_sparsity["CR"]
        Avg_res_sparsity = res_sparsity["Avg"]

    else:
        with open('properties.json', 'r') as file:
            data = json.load(file)
        RC_res_sparsity = truncate_to_four_decimals(data["5"]["sparsity"])
        CR_res_sparsity = truncate_to_four_decimals(data["6"]["sparsity"])
        Avg_res_sparsity = truncate_to_four_decimals(data["7"]["sparsity"])

    result = {
        "RC": {
            "avg_time": RC_avg_matmul_time,
            "sparsity": RC_res_sparsity,
            "operation": RC_mamtmul_operation,
        },
        "CR": {
            "avg_time": CR_avg_matmul_time,
            "sparsity": CR_res_sparsity,
            "operation": CR_mamtmul_operation,
        },
        "Average": {
            "avg_time": Avg_avg_matmul_time,
            "sparsity": Avg_res_sparsity,
            "operation": Avg_mamtmul_operation,
        }
    }
    return result

def plot_results(results, save_path="mnc_combined_performance_plot.png"):
    color_no_args = 'orange'
    color_with_args = 'blue'
    bar_width = 0.35
    inference_labels = ['Estimated Props.', 'Measured Props.']

    fig, (ax_rc, ax_cr, ax_avg) = plt.subplots(1, 3, figsize=(20, 8))
    
    # RC Plot
    result_no_args_rc, result_with_args_rc = results[0]  # Results for RC
    avg_time_no_args_rc, sparsity_no_args_rc, operation_no_args_rc = result_no_args_rc
    avg_time_with_args_rc, sparsity_with_args_rc, operation_with_args_rc = result_with_args_rc

    indices_rc = np.array([0])  # Single bar group for RC

    # Plot RC data
    ax_rc.bar(indices_rc - bar_width/2, avg_time_no_args_rc, bar_width, label='No Args', color=color_no_args)
    ax_rc.bar(indices_rc + bar_width/2, avg_time_with_args_rc, bar_width, label='With Args', color=color_with_args)

    # Add text for RC
    ax_rc.text(indices_rc[0] - bar_width/2, avg_time_no_args_rc, 
               f'Op: {operation_no_args_rc}, Sparsity: {sparsity_no_args_rc:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)
    ax_rc.text(indices_rc[0] + bar_width/2, avg_time_with_args_rc, 
               f'Op: {operation_with_args_rc}, Sparsity: {sparsity_with_args_rc:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)

    # Add label below the bars for RC
    ax_rc.text(indices_rc[0] - bar_width/2, -0.001, 
               inference_labels[0], 
               ha='center', va='top', color='black', fontsize=10)
    ax_rc.text(indices_rc[0] + bar_width/2, -0.001, 
               inference_labels[1], 
               ha='center', va='top', color='black', fontsize=10)

    ax_rc.set_ylabel('Execution Time (s)')
    ax_rc.set_title('RC Operation')
    ax_rc.set_xticks([])  # Remove x-tick labels for bar positions
    ax_rc.set_ylim(0,0.5)
    ax_rc.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    # CR Plot
    result_no_args_cr, result_with_args_cr = results[1]  # Results for CR
    avg_time_no_args_cr, sparsity_no_args_cr, operation_no_args_cr = result_no_args_cr
    avg_time_with_args_cr, sparsity_with_args_cr, operation_with_args_cr = result_with_args_cr

    indices_cr = np.array([0])  # Single bar group for CR

    # Plot CR data
    ax_cr.bar(indices_cr - bar_width/2, avg_time_no_args_cr, bar_width, label='No Args', color=color_no_args)
    ax_cr.bar(indices_cr + bar_width/2, avg_time_with_args_cr, bar_width, label='With Args', color=color_with_args)

    # Add text for CR
    ax_cr.text(indices_cr[0] - bar_width/2, avg_time_no_args_cr + 0.10, 
               f'Op: {operation_no_args_cr}, Sparsity: {sparsity_no_args_cr:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)
    ax_cr.text(indices_cr[0] + bar_width/2, avg_time_with_args_cr + 0.10, 
               f'Op: {operation_with_args_cr}, Sparsity: {sparsity_with_args_cr:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)

    # Add label below the bars for CR
    ax_cr.text(indices_cr[0] - bar_width/2, -0.07, 
               inference_labels[0], 
               ha='center', va='top', color='black', fontsize=10)
    ax_cr.text(indices_cr[0] + bar_width/2, -0.07, 
               inference_labels[1], 
               ha='center', va='top', color='black', fontsize=10)

    ax_cr.set_ylabel('Execution Time (s)')
    ax_cr.set_title('CR Operation')
    ax_cr.set_xticks([])
    ax_cr.set_ylim(0,40)
    ax_cr.yaxis.set_major_locator(ticker.MultipleLocator(5))
    
    # Plot for Average Case
    result_no_args_avg, result_with_args_avg = results[2]  # Results for Average Case
    avg_time_no_args_avg, sparsity_no_args_avg, operation_no_args_avg = result_no_args_avg
    avg_time_with_args_avg, sparsity_with_args_avg, operation_with_args_avg = result_with_args_avg

    indices_avg = np.array([0])  # Single bar group for Average Case

    # Plot Average Case data
    ax_avg.bar(indices_avg - bar_width/2, avg_time_no_args_avg, bar_width, label='No Args', color=color_no_args)
    ax_avg.bar(indices_avg + bar_width/2, avg_time_with_args_avg, bar_width, label='With Args', color=color_with_args)

    # Add text for Average Case
    ax_avg.text(indices_avg[0] - bar_width/2, avg_time_no_args_avg, 
               f'Op: {operation_no_args_avg}, Sparsity: {sparsity_no_args_avg:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)
    ax_avg.text(indices_avg[0] + bar_width/2, avg_time_with_args_avg, 
               f'Op: {operation_with_args_avg}, Sparsity: {sparsity_with_args_avg:.4f}', 
               ha='center', va='bottom', color='black', fontsize=10)

    # Add label below the bars for Average Case
    ax_avg.text(indices_avg[0] - bar_width/2, -0.001, 
               inference_labels[0], 
               ha='center', va='top', color='black', fontsize=10)
    ax_avg.text(indices_avg[0] + bar_width/2, -0.001, 
               inference_labels[1], 
               ha='center', va='top', color='black', fontsize=10)

    ax_avg.set_ylabel('Execution Time (s)')
    ax_avg.set_title('Average Case Operation')
    ax_avg.set_xticks([])  # Remove x-tick labels for bar positions
    ax_avg.set_ylim(0,0.8)
    ax_avg.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    handles, labels = ax_rc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, title="Execution Type", frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(save_path)

    plt.show()

def perform_property_recording(script_path, additional_args=None):
    command = ["bin/daphne"]
    if additional_args:
        command.extend(additional_args)
    command.append("--enable_property_recording")
    command.append(script_path)
    
    subprocess.run(command, capture_output=True, text=True)

def main():
    script_case1_path = "RuntimePropTests/experiment_2_mnc_row_col_nnz/row_col_experiment.daphne"
    iterations = 1

    print("Executing Operation")
    perform_property_recording(script_case1_path)
    print("Property Recording done")
    results_no_args = execute_daphne_script(script_case1_path, iterations=iterations)
    print("Result normal execution done")
    results_with_args = execute_daphne_script(script_case1_path, additional_args=["--enable_property_insert"], iterations=iterations, estimate_res_sparsity=False)
    print("Result Execution with Property Insert done")
    print("Operation finished")

    results = [
        # Results for RC
        (
            (results_no_args["RC"]["avg_time"], results_no_args["RC"]["sparsity"], results_no_args["RC"]["operation"]),
            (results_with_args["RC"]["avg_time"], results_with_args["RC"]["sparsity"], results_with_args["RC"]["operation"])
        ),
        
        # Results for CR
        (
            (results_no_args["CR"]["avg_time"], results_no_args["CR"]["sparsity"], results_no_args["CR"]["operation"]),
            (results_with_args["CR"]["avg_time"], results_with_args["CR"]["sparsity"], results_with_args["CR"]["operation"])
        ),
        
        # Results for Average Case
        (
            (results_no_args["Average"]["avg_time"], results_no_args["Average"]["sparsity"], results_no_args["Average"]["operation"]),
            (results_with_args["Average"]["avg_time"], results_with_args["Average"]["sparsity"], results_with_args["Average"]["operation"])
        )
    ]
    print("Start Plotting")
    plot_results(results)
    print("Plotting done")

if __name__ == "__main__":
    main()