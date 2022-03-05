#!/usr/bin/python

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
# -------------------------------------------------------------

from api.python.context.daphne_context import DaphneContext
import pandas as pd

daphne_context = DaphneContext()

m1 = pd.DataFrame(
    [[21, 72, 68.1],
    [23, 78, 69.5],
    [32, 74, 56.6],
    [52, 54, 86.2]],
    columns=['a', 'b', 'c'])

m2 = pd.DataFrame(
    [[1, 2, 3],
    [4, 5, 6],
    [9, 7, 8],
    [52, 54, 86.2]],
    columns=['d', 'e', 'f'])

matrix1 = daphne_context.from_pandas(m1)
matrix2 = daphne_context.from_pandas(m2)
df = matrix1.cbind(matrix2).compute()
print(df)