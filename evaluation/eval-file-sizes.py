import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt

# Folder containing the CSV files.
data_dir = "data"

# Get all CSV file paths in the data directory.
csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))

results = []

# Helper function to extract a data subtype from the filename.
# Searches for keywords like FLOAT, STR, MIXED (case insensitive).
def extract_subtype(filename):
    subtype = "unknown"
    for candidate in ['float', 'str', 'mixed', 'number', 'rep', 'fixedstr', 'strdiff']:
        if re.search(candidate, filename, re.IGNORECASE):
            subtype = candidate
            break
    return subtype

for csv_file in csv_paths:
    # Get CSV file size in bytes.
    csv_size = os.path.getsize(csv_file)

    # The corresponding dbdf file is assumed to be named as the CSV plus a ".dbdf" extension.
    dbdf_file = csv_file + ".dbdf"
    if os.path.exists(dbdf_file):
        dbdf_size = os.path.getsize(dbdf_file)
    else:
        dbdf_size = None

    # Determine the main data type from the file name.
    base = os.path.basename(csv_file)
    main_type = "matrix" if base.startswith("matrix_") else "frame"

    # Extract data subtype (e.g., float, str, mixed)
    subtype = extract_subtype(base)

    # Compute ratio, if possible (in percentage).
    ratio = (dbdf_size / csv_size * 100) if dbdf_size is not None else None

    results.append({
        "CSVFile": base,
        "MainDataType": main_type,
        "DataSubtype": subtype,
        "CSVSize (bytes)": csv_size,
        "dbdfSize (bytes)": dbdf_size,
        "Ratio (%)": ratio
    })

# Build a DataFrame from the results.
df = pd.DataFrame(results)

# Print the table.
print("Detailed file sizes and ratio:")
print(df)

# Compute the average ratio per MainDataType.
avg_main = df.groupby("MainDataType")["Ratio (%)"].mean().reset_index()
print("\nAverage percentage of dbdf size relative to CSV size by main data type:")
print(avg_main)

# Compute the average ratio per DataSubtype.
avg_subtype = df.groupby("DataSubtype")["Ratio (%)"].mean().reset_index()
print("\nAverage percentage of dbdf size relative to CSV size by data subtype:")
print(avg_subtype)

# Optionally, save the results table to a CSV file.
df.to_csv("file_size_comparison.csv", index=False)
avg_main.to_csv("average_ratio_by_main_type.csv", index=False)
avg_subtype.to_csv("average_ratio_by_subtype.csv", index=False)

# Combine the computed averages with a reference 100% value.
baseline = pd.DataFrame({"Type": ["CSV"], "Ratio (%)": [100], "Group": ["baseline"]})

# avg_main has two rows: one for frame and one for matrix.
avg_main['Group'] = avg_main["MainDataType"]  # "frame" or "matrix"
avg_main = avg_main.rename(columns={"MainDataType": "Type"})

# Compute average ratios by data subtype separately for frame and matrix.
avg_subtype_frame = df[df["MainDataType"] == "frame"].groupby("DataSubtype")["Ratio (%)"].mean().reset_index()
avg_subtype_frame["Group"] = "frame"
avg_subtype_matrix = df[df["MainDataType"] == "matrix"].groupby("DataSubtype")["Ratio (%)"].mean().reset_index()
avg_subtype_matrix["Group"] = "matrix"
avg_subtype_frame = avg_subtype_frame.rename(columns={"DataSubtype": "Type"})
avg_subtype_matrix = avg_subtype_matrix.rename(columns={"DataSubtype": "Type"})

# Concatenate all results.
bar_data = pd.concat([baseline, avg_main, avg_subtype_frame, avg_subtype_matrix], ignore_index=True)

# Now order the bars:
# We'll place baseline first, then frame: first the main frame (i.e. Type=="frame") then its subtype rows sorted alphabetically,
# then matrix: first the main matrix value then its subtype rows sorted alphabetically.
#frame_main = bar_data[(bar_data["Group"]=="frame") & (bar_data["Type"]=="frame")]
frame_sub = bar_data[(bar_data["Group"]=="frame") & (bar_data["Type"]!="frame")].sort_values("Type")
#matrix_main = bar_data[(bar_data["Group"]=="matrix") & (bar_data["Type"]=="matrix")]
matrix_sub = bar_data[(bar_data["Group"]=="matrix") & (bar_data["Type"]!="matrix")].sort_values("Type")

ordered_bar_data = pd.concat([baseline, frame_sub, matrix_sub], ignore_index=True)

# Assign colors: baseline in black, frame group in blue, matrix group in green.
def assign_color(row):
    if row["Group"] == "baseline":
        return "#d62728"  # baseline normal red
    elif row["Group"] == "frame":
        if row["Type"] == "frame":
            return "#1f77b4"  # normal blue for main frame
        else:
            return "#aec7e8"  # light blue for frame subtypes
    elif row["Group"] == "matrix":
        if row["Type"] == "matrix":
            return "#2ca02c"  # normal green for main matrix
        else:
            return "#98df8a"  # light green for matrix subtypes
    else:
        return "gray"

ordered_bar_data["Color"] = ordered_bar_data.apply(assign_color, axis=1)

# Create a bar chart.
plt.figure(figsize=(12,6))
bars = plt.bar(ordered_bar_data["Type"], ordered_bar_data["Ratio (%)"], color=ordered_bar_data["Color"])
plt.xlabel("Data Type / Category")
plt.ylabel("Average dbdf/CSV Size Ratio (%)")
plt.title("Comparison: 100% CSV vs. Average dbdf Ratios by Data Type/Subtype")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig/avg_ratio_bar_chart.png")