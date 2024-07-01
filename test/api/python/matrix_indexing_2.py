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

m1 = daphne_context.fill(123, 10, 10)

m1[0, 0] = daphne_context.fill(1, 1, 1)
m1[0, 0].print().compute()

m1[0:5, 0] = daphne_context.fill(0, 5, 1)
m1[0:5, 0].print().compute()

m1[:5, 0] = daphne_context.fill(1, 5, 1)
m1[:5, 0].print().compute()

m1[0:, 0] = daphne_context.fill(2, 10, 1)
m1[0:, 0].print().compute()

m1[:, 0] = daphne_context.fill(3, 10, 1)
m1[:, 0].print().compute()

m1[0, :] = daphne_context.fill(0, 1, 10)
m1[0, :].print().compute()

m1[0, 0:] = daphne_context.fill(1, 1, 10)
m1[0, 0:].print().compute()

m1[0, :5] = daphne_context.fill(2, 1, 5)
m1[0, :5].print().compute()

m1[0, 0:5] = daphne_context.fill(3, 1, 5)
m1[0, 0:5].print().compute()

m1[:, 0:5] = daphne_context.fill(4, 10, 5)
m1[:, 0:5].print().compute()

m1[0:5, :] = daphne_context.fill(5, 5, 10)
m1[0:5, :].print().compute()

m1.print().compute()

