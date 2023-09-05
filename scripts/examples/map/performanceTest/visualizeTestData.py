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
import json


def createBoxplotsWithSeaborn(csv_file_name):

    data = pd.read_csv(csv_file_name)

    # Unique metric types
    metrics = data['MetricType'].unique()

    # Create a grid of plots
    for metric in metrics:
        g = sns.FacetGrid(data[data['MetricType'] == metric], 
                        row="MatrixSize", 
                        col="Datatype", 
                        hue="Implementation", 
                        height=4, 
                        aspect=2, 
                        margin_titles=True)
    
        # Boxplots for each combination of MatrixSize and Datatype
        g = (g.map(sns.boxplot, "Operation", "Value", order=None)
            .add_legend()
            .fig.suptitle(f'Performance Metrics for {metric} by Matrix Size and Datatype'))
    
        plt.subplots_adjust(top=0.9, hspace=0.3)
        plt.savefig(f"scripts/examples/map/testdata/plots/{csv_file_name}")


def createInteractiveBoxplot(csv_file_name):

    data = pd.read_csv(csv_file_name)
    
    # Interactive Boxplots
    fig = px.box(data, 
                x="Operation", 
                y="Value", 
                color="Implementation", 
                facet_row="Datatype",
                title="Performance Metrics by Operation and Implementation")

    # Add dropdown for metric selection
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

    # Add slider for matrix size selection
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
    with open(f"scripts/examples/map/testdata/plots/{csv_file_name}.json", 'w') as f:
        json.dump(fig.to_dict(), f)