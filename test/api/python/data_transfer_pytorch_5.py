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

# Data transfer from PyTorch to DAPHNE and back, via shared memory.

import torch
from daphne.context.daphne_context import DaphneContext

csc_indptr = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
csc_indices = torch.tensor([2, 2], dtype=torch.int64)
values = torch.tensor([0.962463, 0.28892], dtype=torch.float64)
shape = (3, 3)

csc = torch.sparse_csc_tensor(csc_indptr, csc_indices, values, size=(3,3))

dctx = DaphneContext()

dctx.from_pytorch(csc, shared_memory=True).print().compute(type="shared memory")