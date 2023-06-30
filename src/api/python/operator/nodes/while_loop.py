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
from api.python.operator.nodes.matrix import Matrix
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
import numpy as np
import textwrap

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

from api.python.utils import analyzer

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class WhileLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', pred, callback,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'WhileLoop':
        if analyzer.get_number_argument(pred) != analyzer.get_number_argument(callback):
            raise ValueError(f"{pred} and {callback} do not have the same number of arguents")
        elif analyzer.get_number_argument(callback) != len(unnamed_input_nodes):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes")

        self._callback = callback(*unnamed_input_nodes)
        if (not isinstance(self._callback, tuple)):
            self._callback = (self._callback, )

        self._pred = pred(*unnamed_input_nodes)
        if (not isinstance(self._pred, tuple)):
            self._pred = (self._pred, )

        if len(self._callback) != len(unnamed_input_nodes):
            raise ValueError(f"{callback} and does not have the same number return values as input nodes")
        elif len(self._pred) != 1:
            raise ValueError(f"{pred} do not have the 1 return value, but that is required")

        _named_input_nodes = dict()

        outer_vars_pred = analyzer.get_outer_scope_variables(pred)
        for i, var in enumerate(outer_vars_pred):
            inside_node = pred.__globals__.get(var)
            if var:
                _named_input_nodes.update({f"p_var{i}": inside_node})

        outer_vars = analyzer.get_outer_scope_variables(callback)
        for i, var in enumerate(outer_vars):
            inside_node = callback.__globals__.get(var)
            if var:
                _named_input_nodes.update({f"var{i}": inside_node})

        self.copy = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, " + ", [node, 0])
            new_matrix_node._source_node = self
            self.copy.append(new_matrix_node)

        super().__init__(daphne_context, 'while_loop', unnamed_input_nodes=unnamed_input_nodes,
            named_input_nodes=_named_input_nodes, output_type=OutputType.NONE)

    def get_copy(self):
        return tuple(self.copy)
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:

        parent_level = 1  # default
        if self._script:
            parent_level = self._script._nested_level

        callback_script = NestedDaphneDSLScript(self.daphne_context, parent_level+1)
        callback_names = callback_script.build_code(self._callback)
        
        pred_script = NestedDaphneDSLScript(self.daphne_context, parent_level + 1, 'C')
        pred_name = pred_script.build_code(self._pred)[0]

        callback_body = callback_script.daphnedsl_script
        for i, name in enumerate(callback_names):
            callback_body += f"{unnamed_input_vars[i]}={name};\n"

        pred_body = pred_script.daphnedsl_script

        multiline_str = str()
        multiline_str += pred_body
        multiline_str += f"while ({pred_name}) {{\n"
        multiline_str += textwrap.indent(callback_body, prefix="    ")
        multiline_str += textwrap.indent(pred_body, prefix="    ")
        multiline_str += "}"

        return multiline_str
    
        # multiline_str = ""
        # multiline_str += f"{self._pred._script.daphnedsl_script}"
        # multiline_str += f"while ({self._pred.daphnedsl_name}) {{\n"
        # #multiline_str += f"    {named_input_vars['number']} = {named_input_vars['node']} + {self.daphnedsl_name};
        # multiline_str += textwrap.indent(self._callback._script.daphnedsl_script, prefix="    ")
        # multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._callback.daphnedsl_name};\n", prefix="    ")
        # multiline_str += textwrap.indent(f"{self._pred._script.daphnedsl_script}", prefix="    ")
        # multiline_str += "}"
        # self.daphnedsl_name = named_input_vars['node']
        # try

    def compute(self) -> None:
        return super().compute()