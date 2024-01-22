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

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([1,  2, 3,  4], dtype=np.double)
m2 = np.array([1, -1, 1, -1], dtype=np.double)
m1.shape = (2, 2)
m2.shape = (2, 2)
dctx = DaphneContext()

res = (dctx.from_numpy(m1) @ dctx.from_numpy(m2)).compute()
print(res.sum())

res = (dctx.from_numpy(m1) < dctx.from_numpy(m2)).compute()
print(res.sum())

res = (dctx.from_numpy(m1) > dctx.from_numpy(m2)).compute()
print(res.sum())

res = (dctx.from_numpy(m1) == dctx.from_numpy(m2)).compute()
print(res.sum())