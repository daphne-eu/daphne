import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Folder where logs are stored.
results_dir = './results'

# This function extracts dimensions (number of rows and columns) from the filename.
# e.g. "frame_100000r_20c_MIXED.csv" -> (100000,20)
def extract_dims(filename):
    m = re.search(r'(\d+)r_(\d+)c', filename)
    if m:
        rows = int(m.group(1))
        cols = int(m.group(2))
        return rows, cols
    else:
        return None, None

# This function extracts the overall data type from the filename.
# It considers the main type (matrix if the filename starts with "matrix_",
# otherwise frame) combined with a subtype (mixed, str, float, etc.).
def extract_data_type(filename):
    base = os.path.basename(filename)
    main_type = "matrix" if base.startswith("matrix_") else "frame"
    m = re.search(r'(mixed|str|float|rep|strdiff|fixedstr|number)', base, re.IGNORECASE)
    subtype = m.group(1).lower() if m else "unknown"
    # Map fixedstr and strdiff to "str" for comparison purposes
    if subtype in ["fixedstr", "strdiff"]:
        subtype = "str"
    return f"{main_type}_{subtype}"

# Load CSV logs for each experiment.
def load_log(experiment, pattern):
    # We assume files are named like evaluation_results_*_{experiment}.csv in the results folder.
    files = glob.glob(os.path.join(results_dir, f"evaluation_results_*_{experiment}.csv"))
    dfs = []
    for f in files:
        # The CSV already has a header:
        # CSVFile,Experiment,Trial,ReadTime,WriteTime,dbdfReadTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds
        df = pd.read_csv(f)
        # Extract dimensions and add them as columns.
        dims = df['CSVFile'].apply(lambda x: extract_dims(x))
        df['Rows'] = dims.apply(lambda x: x[0] if x else np.nan)
        df['Cols'] = dims.apply(lambda x: x[1] if x else np.nan)
        # Compute a size measure (for example, total cells)
        df['Size'] = df['Rows'] * df['Cols']
        # Extract a combined data type (main type and subtype).
        df['DataType'] = df['CSVFile'].apply(extract_data_type)
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Load the three experiment logs.
df_normal = load_log("normal", "evaluation_results_*_normal.csv")
df_create = load_log("create", "evaluation_results_*_create.csv")
df_opt = load_log("opt", "evaluation_results_*_opt.csv")

# Compute average timings per dataset (grouped by CSVFile, Size, Rows, Cols, and DataType)
def aggregate_log(df):
    # Convert timing fields to numeric type.
    cols_to_numeric = ['ReadTime', 'WriteTime',
                       'StartupSeconds', 'ParsingSeconds', 'CompilationSeconds',
                       'ExecutionSeconds', 'TotalSeconds']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Group including DataType so that it is preserved in the aggregation.
    return df.groupby(['CSVFile', 'Size', 'Rows', 'Cols', 'DataType'])[cols_to_numeric].mean().reset_index()

agg_normal = aggregate_log(df_normal)
agg_create = aggregate_log(df_create)
agg_opt = aggregate_log(df_opt)

# Plot 1: Overall read time comparison for Normal, First (Create) and Second (Opt) reads.
plt.figure(figsize=(10,6))
agg_normal = agg_normal.sort_values("Size")
agg_create = agg_create.sort_values("Size")
agg_opt = agg_opt.sort_values("Size")

plt.plot(agg_normal["Size"], agg_normal["ReadTime"], marker="o", label="Normal Read")
plt.plot(agg_create["Size"], agg_create["ReadTime"], marker="s", label="First Read (Overall)")
plt.plot(agg_opt["Size"], agg_opt["ReadTime"], marker="^", label="Second Read (Overall)")
plt.xlabel("Dataset Size (Rows x Cols)")
plt.ylabel("Overall Read Time (seconds)")
plt.title("Overall Read Time vs Dataset Size")
plt.xscale("log")  # Added: logarithmic scale on x-axis.
plt.yscale("log")  # Added: logarithmic scale on y-axis.
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("/fig/overall_read_time.png")
plt.close()

# Plot 2: Three read comparison per dataset size for each data type.
unique_types = agg_normal["DataType"].unique()
for dt in unique_types:
    sub_normal = agg_normal[agg_normal["DataType"] == dt].sort_values("Size")
    sub_create = agg_create[agg_create["DataType"] == dt].sort_values("Size")
    sub_opt = agg_opt[agg_opt["DataType"] == dt].sort_values("Size")

    plt.figure(figsize=(10,6))
    plt.plot(sub_normal["Size"], sub_normal["ReadTime"], marker="o", label="Normal Read")
    plt.plot(sub_create["Size"], sub_create["ReadTime"], marker="s", label="First Read (Overall)")
    plt.plot(sub_opt["Size"], sub_opt["ReadTime"], marker="^", label="Second Read (Overall)")
    plt.xlabel("Dataset Size (Rows x Cols)")
    plt.ylabel("Overall Read Time (seconds)")
    plt.title(f"Overall Read Time vs Dataset Size for {dt}")
    plt.xscale("log")  # Added: logarithmic scale on x-axis.
    plt.yscale("log")  # Added: logarithmic scale on y-axis.
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f"/fig/overall_read_time_{dt}.png")
    plt.close()
    
# Plot 3: Breakdown for First Read (Create) – Stacked bar: Overall Read Time and dbdf Write Time.
if not agg_create.empty:
    ind = np.arange(len(agg_create))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10,6))
    p1 = ax.bar(ind, agg_create["ReadTime"], width, label="Overall Read Time")
    p2 = ax.bar(ind, agg_create["WriteTime"], width, bottom=agg_create["ReadTime"], label="dbdf Write Time")
    ax.set_xticks(ind)
    ax.set_xticklabels(agg_create["CSVFile"], rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("First Read Breakdown (Create): Read vs. Write dbdf")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/fig/create_read_breakdown.png")
    plt.close()

# Plot 4: Breakdown for Second Read (Opt) – Stacked bar: dbdf Read Time and Overall Read Time.
if not agg_opt.empty:
    ind = np.arange(len(agg_opt))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10,6))
    p1 = ax.bar(ind, agg_opt["dbdfReadTime"], width, label="dbdf Read Time")
    p2 = ax.bar(ind, agg_opt["ReadTime"], width, bottom=agg_opt["dbdfReadTime"], label="Overall Read Time")
    ax.set_xticks(ind)
    ax.set_xticklabels(agg_opt["CSVFile"], rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Second Read Breakdown (Opt): dbdf vs. Overall Read")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/fig/opt_read_breakdown.png")
    plt.close()

print("Charts generated and saved as PNG files.")