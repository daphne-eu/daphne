# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api.python.context.daphne_context import DaphneContext
import pandas as pd
import numpy as np
import csv
import time

dc = DaphneContext()

#Create Lists for the PerfoamnceResults Dataframe 
type = []
dataframe_size = []
transfer_daphne_time = []
rbind_time = []
cartesian_time = []

# Creating a list of sizes for the objects
sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]


# DATAFRAME 
# Creating a list of dataframes with different sizes
dataframes = [pd.DataFrame(np.random.randn(size, 15)) for size in sizes]
# Looping through the dataframes and testing the from_pandas and compute operation
print("\n\n###\n### Dataframes Experiments:\n###\n")
for df in dataframes:
    type.append("Dataframe")

    print("Dataframe Size:",df.size)
    dataframe_size.append(df.size)

    # Transfer data to DaphneLib
    start_time = time.time()
    F = dc.from_pandas(df, verbose=True)
    end_time = time.time()
    transfer_daphne_time.append(end_time-start_time)

    # Appending and cartesian calculation
    r_start_time = time.time()
    F = F.rbind(F)
    F.compute(verbose=True)
    r_end_time = time.time()
    rbind_time.append(r_end_time-r_start_time)
    
    c_start_time = time.time()
    F = F.cartesian(F)
    F.compute(verbose=True)
    c_end_time = time.time()
    rbind_time.append(c_end_time-c_start_time)
    print(rbind_time)

    print(F.compute())
print("\n###End of Dataframe Experiments.\n")

# SERIES
# Creating a list of series with different sizes
series = [pd.Series(np.random.randn(size)) for size in sizes]
# Looping through the series and testing the from_pandas and compute operation
print("\n\n###\n### Series Experiments:\n###\n")
for s in series:
    print("Series Size:",s.size)
    # Transfer data to DaphneLib
    F = dc.from_pandas(s, verbose=True)
    # Appending and cartesian calculation
    F = F.rbind(F)
    F.compute()
    F = F.cartesian(F)
    F.compute()
    print(F.compute())
print("\n###End of Series Experiments.\n")

# SPARSE DATAFRAME
# Creating a list of sparse dataframes with different sizes
sparse_dataframes = [pd.SparseDataFrame(np.random.randn(size, 15)) for size in sizes]
# Looping through the sparse dataframes and testing the from_pandas and compute operation
print("\n\n###\n### Sparse DataFrame Experiments:\n###\n")
for sdf in sparse_dataframes:
    print("Sparse Dataframe Size:",sdf.size)
    # Transfer data to DaphneLib
    F = dc.from_pandas(sdf, verbose=True)
    # Appending and cartesian calculation
    F = F.rbind(F)
    F.compute()
    F = F.cartesian(F)
    F.compute()
    print(F.compute())
print("\n###End of Sparse DataFrame Experiments.\n")

# MULTIINDEX DATAFRAME
# Creating a list of multiindex dataframes with different sizes
multiindex_dataframes = [pd.MultiIndex.from_product([list('"AB"'), range(size)]) for size in sizes]
# Looping through the categorical dataframes and testing the from_pandas and compute operation
print("\n\n###\n### MultiIndex Experiments:\n###\n")
for midf in multiindex_dataframes:
    # Transfer data to DaphneLib
    F = dc.from_pandas(midf, verbose=True)
    # Appending and cartesian calculation
    F = F.rbind(F)
    F.compute()
    F = F.cartesian(F)
    F.compute()
    print(F.compute())
print("\n###End of MultiIndex Experiments.\n")

# CATEGORICAL DATAFRAME
# Creating a list of categorical dataframes with different sizes
categorical_dataframes = [pd.DataFrame(np.random.randn(size, 15)) for size in sizes]
# Looping through the multiindex and testing the from_pandas and compute operation
print("\n\n###\n### Categorical Dataframe Experiments:\n###\n")
for cdf in categorical_dataframes:
    cdf = cdf.astype("category")
    # Transfer data to DaphneLib
    F = dc.from_pandas(cdf, verbose=True)
    # Appending and cartesian calculation
    F = F.rbind(F)
    F.compute()
    F = F.cartesian(F)
    F.compute()
    print(F.compute())
print("\n###End of Categorical DataFrame Experiments.\n")

#Create PerfoamnceResults Dataframe from Lists and export as CSV
zipped = list(zip(type, dataframe_size, transfer_daphne_time, rbind_time, cartesian_time))
PerfoamnceResults = pd.DataFrame(zipped, columns=['Type', 'dataframe_size', 'transfer_daphne_time', 'rbind_time', 'cartesian_time'])

#PerfoamnceResults.to_csv("scripts/examples/daphnelib/results.csv")
print(PerfoamnceResults)
#print(type)
#print(dataframe_size)
# print(transfer_daphne_time)