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

def createBoxplotsWithSeaborn(csv_file_name):
    data = pd.read_csv(csv_file_name)

    metrics = data['MetricType'].unique()

    for metric in metrics:
        subset_data = data[data['MetricType'] == metric]
        
        unique_sizes = subset_data['Matrix Size'].unique()
        unique_dtypes = subset_data['Datatype'].unique()

        for size in unique_sizes:
            for dtype in unique_dtypes:
                plt.figure(figsize=(8, 6))
                specific_data = subset_data[(subset_data['Matrix Size'] == size) & (subset_data['Datatype'] == dtype)]
                
                if specific_data.empty:
                    print(f"No data for Metric: {metric}, Matrix Size: {size}, Datatype: {dtype}")
                    continue
                if len(specific_data['Implementation'].unique()) <= 1:
                    print(f"Only one unique hue for Metric: {metric}, Matrix Size: {size}, Datatype: {dtype}")
                    continue
                specific_data = specific_data.dropna()
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

    matrix_sizes = sorted(data['Matrix Size'].unique())
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
                "prefix": "Matrix Size:",
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
    #createInteractiveBoxplot("scripts/examples/map/performanceTest/testdata/csv_files/performance_results_2023-09-11 01:07:39.csv")
    createBoxplotsWithSeaborn("scripts/examples/map/performanceTest/testdata/csv_files/performance_results_2023-09-11 01:07:39.csv")