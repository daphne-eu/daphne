import subprocess
import matplotlib.pyplot as plt

def execute_daphne_script(script_path, additional_args=None, iterations=10):
    command = ["bin/daphne"]
    if additional_args:
        command.extend(additional_args)
    
    command.append(script_path)
        
    times = []
    for _ in range(iterations):
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout
        
        for line in output.splitlines():
            if "timestamp:" in line:
                timestamp_ns = int(line.split("timestamp:")[1].strip())
                times.append(timestamp_ns)
    
    print(f"Times for script {script_path}: {times}")

    avg_time = sum(times) / iterations
    
    return avg_time

def plot_results(results, save_path="property_recording_plot.png"):
    (op1_no_args, op2_no_args), (op1_with_args, op2_with_args) = results
    
    operations = ['Operation 1', 'Operation 2']
    no_args = [op1_no_args, op2_no_args]
    with_args = [op1_with_args, op2_with_args]

    x = range(len(operations))
    bar_width = 0.4

    fig, ax = plt.subplots()
    ax.bar(x, no_args, width=bar_width, label='Daphne Inference', align='center')
    ax.bar(x, with_args, width=bar_width, label='Property Recording Inference', align='edge')

    ax.set_xlabel('Operations')
    ax.set_ylabel('Time (ns)')
    ax.set_title('Execution Time of Daphne Script')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()

    plt.savefig(save_path)
    plt.close()

def perform_property_recording(script_path, additional_args=None):
    command = ["bin/daphne"]
    command.append("--enable_property_recording")
    command.append(script_path)
    subprocess.run(command, capture_output=True, text=True)

def main():
    script_case1_path = "RuntimePropTests/RuntimePropTest_case1.daphne"
    script_case2_path = "RuntimePropTests/RuntimePropTest_case2.daphne"
    iterations = 10
    warmup_iterations = 5
    
    perform_property_recording(script_case1_path)
    op1_no_args = execute_daphne_script(script_case1_path, iterations=iterations)
    op1_with_args = execute_daphne_script(script_case1_path, additional_args=["--enable_property_insert"], iterations=iterations)

    perform_property_recording(script_case2_path)
    op2_no_args = execute_daphne_script(script_case2_path, iterations=iterations)
    op2_with_args = execute_daphne_script(script_case2_path, additional_args=["--enable_property_insert"], iterations=iterations)

    results = ((op1_no_args, op2_no_args), (op1_with_args, op2_with_args))

    plot_results(results)

if __name__ == "__main__":
    main()
