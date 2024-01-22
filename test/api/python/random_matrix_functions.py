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


from daphne.context.daphne_context import DaphneContext


daphne_context = DaphneContext()

m1 = daphne_context.rand(rows=5,cols=5,min=1.0,max=5.0,sparsity=1,seed=123)
m2 = daphne_context.rand(rows=5,cols=5,min=1.0,max=105.0,sparsity=1,seed=123)

m1.max(m2).print().compute()
m1.sqrt().print().compute()
m1.min(m2).print().compute()