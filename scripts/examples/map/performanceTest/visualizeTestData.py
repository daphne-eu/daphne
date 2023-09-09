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

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio

def visualize_performance_metrics_datatypes_summarized(csv_file_name):
    df = pd.read_csv(csv_file_name)
    
    for metric in df['MetricType'].unique():
        for size in df['MatrixSize'].unique():
            subset_data = df[(df['MetricType'] == metric) & (df['MatrixSize'] == size)]
            
            if not subset_data.empty:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=subset_data, x='Operation', y='Value', hue='Implementation', palette="Set3", width=0.6)
                
                sns.swarmplot(data=subset_data, x='Operation', y='Value', hue='Implementation', size=2, dodge=True, linewidth=0.5, edgecolor='gray', palette="dark")
                
                plt.title(f'Boxplot of {metric} by Implementation (Matrix Size: {size})')
                plt.ylabel(metric)
                plt.xlabel('Operation')
                plt.legend(title='Implementation')
                plt.tight_layout()
                    
                # Construct save file name
                save_file_name = f"{csv_file_name.split('.')[0]}_{metric}_{size}.png"
                plt.savefig(save_file_name)
                plt.close()

def createBoxplotsWithSeaborn(csv_file_name):
    data = pd.read_csv(csv_file_name)

    metrics = data['MetricType'].unique()

    for metric in metrics:
        subset_data = data[data['MetricType'] == metric]
        
        unique_sizes = subset_data['MatrixSize'].unique()
        unique_dtypes = subset_data['Datatype'].unique()

        for size in unique_sizes:
            for dtype in unique_dtypes:
                plt.figure(figsize=(8, 6))
                specific_data = subset_data[(subset_data['MatrixSize'] == size) & (subset_data['Datatype'] == dtype)]
                sns.boxplot(data=specific_data, x='Operation', y='Value', hue='Implementation')
                
                plt.title(f'Metric: {metric}, Matrix Size: {size}, Datatype: {dtype}')
                
                save_file_name = f"{csv_file_name.split('.')[0]}_{metric}_{size}_{dtype}.png"
                plt.savefig(save_file_name)
                plt.close()

def createInteractiveBoxplot(csv_file_name):

    data = pd.read_csv(csv_file_name)
    
    fig = px.box(data, 
                x="Operation", 
                y="Value", 
                color="Implementation", 
                facet_row="Datatype",
                title="Performance Metrics by Operation and Implementation")

    metric_dropdown = [
        {
            "label": metric,
            "method": "restyle",
            "args": [
                {
                    "visible": [metric == m for m in data["MetricType"].unique()],
                    "y": [data[data["MetricType"] == metric]["Value"] if metric == m else None for m in data["MetricType"].unique()],
                }
            ]
        }
        for metric in data["MetricType"].unique()
    ]

    matrix_sizes = sorted(data['MatrixSize'].unique())
    fig.update_layout(
        updatemenus=[
            {
                "buttons": metric_dropdown,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.2,
                "yanchor": "top"
            },
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "MatrixSize:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        {"frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}},
                        {"frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}}
                    ],
                    "label": str(size),
                    "method": "animate",
                    "value": str(size)
                }
                for size in matrix_sizes
            ]
        }]
    )
    fig.show()
    pio.write_html(fig, file=f"{csv_file_name}.html")

if __name__ == "__main__":
    createInteractiveBoxplot("scripts/examples/map/performanceTest/testdata/csv_files/performance_results_2023-09-08 12:04:23.csv")
    #visualize_performance_metrics_datatypes_summarized("scripts/examples/map/performanceTest/testdata/csv_files/performance_results_2023-09-08 12:04:23.csv")