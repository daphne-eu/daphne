#!/usr/bin/python

# Copyright 2023 The DAPHNE Consortium
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

# Data transfer from pandas to DAPHNE and back, via shared memory.
# pd.DataFrame with sparse data 

import pandas as pd
from daphne.context.daphne_context import DaphneContext

sdf = pd.DataFrame({
    "A": pd.arrays.SparseArray([1, 0, 0]),
    "B": pd.arrays.SparseArray([0, 2, 0]),
    "C": pd.arrays.SparseArray([0, 0, 3])
})

dctx = DaphneContext()

dctx.from_pandas(sdf, shared_memory=True).print().compute(type="shared memory")