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
from api.python.context.daphne_context import DaphneContext

dim = 5
m1 = np.array(np.random.randint(10, size=dim*dim), dtype=np.double)
m2 = np.array(np.random.randint(10, size=dim*dim), dtype=np.double)
m1.shape=(dim,dim)
m2.shape=(dim,dim)
daphne_context = DaphneContext()


result = (daphne_context.from_numpy(m1)@daphne_context.from_numpy(m2)).compute()
print(result.sum())

result = (daphne_context.from_numpy(m1)<daphne_context.from_numpy(m2)).compute()
print(result.sum())

result = (daphne_context.from_numpy(m1)>daphne_context.from_numpy(m2)).compute()
print(result.sum())

result = (daphne_context.from_numpy(m1)==daphne_context.from_numpy(m2)).compute()
print(result.sum())

