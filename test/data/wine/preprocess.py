#!/usr/bin/env python3

# Copyright 2024 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Downloads the Wine Quality data set (https://archive.ics.uci.edu/dataset/186/wine+quality)
and preprocesses it to create a combined CSV file for red and white wines.
"""

import json
import math
import os
import pandas as pd
from urllib.request import urlretrieve
import zipfile

# Define file names.
NAME_RED = "winequality-red.csv"
NAME_WHT = "winequality-white.csv"
NAME_REDWHT = "winequality-red-white.csv"

# Download the data set ZIP and extract the CSV files.
zipPath, httpHeaders = urlretrieve(
    "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "wine+quality.zip"
)
with zipfile.ZipFile(zipPath, "r") as z:
    z.extractall(".", [NAME_RED, NAME_WHT])

# Read the two separate data sets.
dfRed = pd.read_csv(NAME_RED, sep=";")
dfWht = pd.read_csv(NAME_WHT, sep=";")

# Append the class label.
dfRed["class"] = 1 # red wine
dfWht["class"] = 2 # white wine

# Concatenate the two data sets.
dfRedWht = pd.concat([dfRed, dfWht])

# Write the combined data set.
dfRedWht.to_csv(
    NAME_REDWHT,
    sep=",", header=False, index=False,
    # Don't write trailing ".0" of integer values in float columns (to save some space).
    float_format=lambda x: x if math.ceil(x) != x else str(int(x))
)

# Generate the meta data file for DAPHNE.
def mapDType(colType):
    if pd.api.types.is_float_dtype(colType):
        return "f64"
    if pd.api.types.is_int64_dtype(colType):
        return "si64"
    else:
        raise RuntimeError(f"unknown pandas dtype: '{colType}'")
meta = {
    "numRows": len(dfRedWht),
    "numCols": len(dfRedWht.columns),
    "schema": [
        {"label": cn, "valueType": mapDType(ct)}
        for cn, ct in zip(dfRedWht.columns, dfRedWht.dtypes)
    ]
}
with open(f"{NAME_REDWHT}.meta", "w") as f:
    json.dump(meta, f, indent=2)

# Tidy up.
os.remove(zipPath)
os.remove(NAME_RED)
os.remove(NAME_WHT)