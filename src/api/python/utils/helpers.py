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
from itertools import chain
from typing import Dict, Iterable, Sequence
from api.python.utils.consts import VALID_INPUT_TYPES

def create_params_string(unnamed_parameters: Iterable[str], named_parameters:Dict[str,str])->str:
    named_input_strs=(f'{v}'for (k,v) in named_parameters.items())
    return ','.join(chain(unnamed_parameters, named_input_strs))

def get_slice_string(i):
    if isinstance(i, tuple):
        if len(i) > 2:
            raise ValueError(
                f'Invalid number of dimensions to slice {len(i)}, Only 2 dimensions allowed')
        else:
            return f'{get_slice_string(i[0])},{get_slice_string(i[1])}'
    elif isinstance(i, slice):
        if i.step:
            raise ValueError("Invalid to slice with step in systemds")
        elif i.start == None and i.stop == None:
            return ''
        elif i.start == None or i.stop == None:
            raise NotImplementedError("Not Implemented slice with dynamic end")
        else:
            # + 1 since R and systemDS is 1 indexed.
            return f'{i.start+1}:{i.stop}'
    else:
        # + 1 since R and systemDS is 1 indexed.
        sliceIns = i+1
    return sliceIns