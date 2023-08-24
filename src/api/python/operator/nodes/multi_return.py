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

from api.python.operator.operation_node import OperationNode
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES, BINARY_OPERATIONS
from api.python.utils.helpers import create_params_string

import numpy as np

from typing import Union, TYPE_CHECKING, Dict, Iterable, Optional, Sequence

class MultiReturn(OperationNode):
    _outputs: Iterable['OperationNode']

    def __init__(self, daphne_context, operation:str, output_nodes: Iterable[VALID_INPUT_TYPES], unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]]=None, 
                named_input_nodes:Dict[str, VALID_INPUT_TYPES]=None):
        self._outputs = output_nodes
        
        for node in self._outputs:
            node._source_node = self
        super().__init__(daphne_context, operation, unnamed_input_nodes, named_input_nodes, OutputType.MULTI_RETURN, False)
    
    def __getitem__(self, index):
        return self._outputs[index]

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        inputs_comma_sep = create_params_string(
            unnamed_input_vars, named_input_vars)
        output_list=list()
        for idx, output_node in enumerate(self._outputs):
            name = f'{var_name}_{idx}'
            output_node.daphnedsl_name = name
            output_list.append(name)
        output_str = ",".join(output_list)
        return f'{output_str}={self.operation}({inputs_comma_sep});'

