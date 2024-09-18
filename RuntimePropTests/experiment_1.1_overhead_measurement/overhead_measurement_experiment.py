import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def run_daphne_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = result.stdout
    
    for line in output.splitlines():
        if line.startswith("{"):
            timing_info = json.loads(line)
            return timing_info
    
    raise ValueError("No JSON found in the command output")

def plot_timing_comparison(without_property_data, with_property_data, save_path="overhead_measurement_experiment_bar_plot.png"):
    labels = ['Startup', 'Parsing', 'Compilation', 'Execution']

    without_property_times = [
        without_property_data['startup_seconds'],
        without_property_data['parsing_seconds'],
        without_property_data['compilation_seconds'],
        without_property_data['execution_seconds']
    ]

    with_property_times = [
        with_property_data['startup_seconds'],
        with_property_data['parsing_seconds'],
        with_property_data['compilation_seconds'],
        with_property_data['execution_seconds']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(labels))

    p1 = ax.bar(index, without_property_times, bar_width, label='Estimated Props.')
    p2 = ax.bar(index + bar_width, with_property_times, bar_width, label='Measured Props.')

    for i, rect in enumerate(p1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{without_property_times[i]:.3f}s', ha='center', va='bottom')

    for i, rect in enumerate(p2):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{with_property_times[i]:.3f}s', ha='center', va='bottom')

    ax.set_xlabel('Operations')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Daphne Execution Timing Phases')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_stacked_timing_comparison(without_property_data, with_property_data, save_path="overhead_measurement_experiment_stacked_plot.png"):
    labels = ['Estimated Props.', 'Measured Props.']
    phases = ['startup_seconds', 'parsing_seconds', 'compilation_seconds', 'execution_seconds']
    phase_colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']

    without_property_times = [
        without_property_data['startup_seconds'],
        without_property_data['parsing_seconds'],
        without_property_data['compilation_seconds'],
        without_property_data['execution_seconds']
    ]

    with_property_times = [
        with_property_data['startup_seconds'],
        with_property_data['parsing_seconds'],
        with_property_data['compilation_seconds'],
        with_property_data['execution_seconds']
    ]

    index = np.arange(2)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.bar(index[0], without_property_times[0], bar_width, label='Startup', color=phase_colors[0])
    ax.bar(index[0], without_property_times[1], bar_width, bottom=without_property_times[0], label='Parsing', color=phase_colors[1])
    ax.bar(index[0], without_property_times[2], bar_width, bottom=np.add(without_property_times[0], without_property_times[1]), label='Compilation', color=phase_colors[2])
    ax.bar(index[0], without_property_times[3], bar_width, bottom=np.add(np.add(without_property_times[0], without_property_times[1]), without_property_times[2]), label='Execution', color=phase_colors[3])

    ax.bar(index[1], with_property_times[0], bar_width, color=phase_colors[0])
    ax.bar(index[1], with_property_times[1], bar_width, bottom=with_property_times[0], color=phase_colors[1])
    ax.bar(index[1], with_property_times[2], bar_width, bottom=np.add(with_property_times[0], with_property_times[1]), color=phase_colors[2])
    ax.bar(index[1], with_property_times[3], bar_width, bottom=np.add(np.add(with_property_times[0], with_property_times[1]), with_property_times[2]), color=phase_colors[3])

    ax.set_xlabel('Execution Mode')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Daphne Execution Timing Phases')
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    
    ax.set_ylim(0,6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) 

    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def main():
    without_property_cmd= ["bin/daphne","--timing", "RuntimePropTests/experiment_1.1_overhead_measurement/overhead_measurement_experiment.daphne"]
    with_property_cmd= ["bin/daphne","--timing", "--enable_property_recording", "RuntimePropTests/experiment_1.1_overhead_measurement/overhead_measurement_experiment.daphne"]

    print("Running without property recording...")
    without_property_data = run_daphne_command(without_property_cmd)

    print("Running with property recording...")
    with_property_data = run_daphne_command(with_property_cmd)

    plot_stacked_timing_comparison(without_property_data, with_property_data)

if __name__ == "__main__":
    main()
