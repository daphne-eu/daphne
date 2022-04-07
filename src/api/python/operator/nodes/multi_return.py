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
# Modifications Copyright 2022 The DAPHNE Consortium
#
# -------------------------------------------------------------
from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS
from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.scalar import Scalar
import numpy as np

from helpers import create_params_string

class MultiReturn(OperationNode):
    _np_array: np.array

    def __init__(self, operation:str,output_nodes, unnamed_input_nodes:Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None):
        self._outputs = output_nodes
        super().__init__(operation, unnamed_input_nodes, named_input_nodes, OutputType.MULTI_RETURN, False)
    
    def __getitem__(self, key):
        self._outputs[key]

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        inputs_comma_sep = create_params_string(
            unnamed_input_vars, named_input_vars)
        output="["
        for idx, output_node in enumerate(self._outputs):
            name = f'{var_name}_{idx}'
            output_node.daphnedsl_name = name
            output += f'{name},'
        output = output[:-1]+"]"
        return f'{output}={self.operation}({inputs_comma_sep})'

