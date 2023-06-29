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
                 named_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'WhileLoop':
        _named_input_nodes = named_input_nodes

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

        self._pred = pred(named_input_nodes['node'])
        #_named_input_nodes.update({"cond": self._pred})

        super().__init__(daphne_context, 'while_loop', named_input_nodes=named_input_nodes,
                         output_type=OutputType.MATRIX)
        
        
        self._callback = callback(named_input_nodes['node'])
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:


        parent_level = self._script._nested_level

        self._callback._script = NestedDaphneDSLScript(self._callback.daphne_context, parent_level+1)
        self._callback._script.build_code(self._callback)
        
        self._pred._script = NestedDaphneDSLScript(self._pred.daphne_context, parent_level + 1, 'C')
        self._pred._script.build_code(self._pred)

        # script = NestedDaphneDSLScript(self.daphne_context, parent_level+1)
        # script.build_code(self._callback)
        # print(f"callback script: {script.daphnedsl_script}")
        # script.daphnedsl_script = ""
        # script.build_code(self._pred)
        # print(f"pred script: {script.daphnedsl_script}")
        #print(self._callback._script.daphnedsl_script)
        #print(self._pred._script.daphnedsl_script)

        multiline_str = ""
        multiline_str += f"{self._pred._script.daphnedsl_script}"
        multiline_str += f"while ({self._pred.daphnedsl_name}) {{\n"
        #multiline_str += f"    {named_input_vars['number']} = {named_input_vars['node']} + {self.daphnedsl_name};
        multiline_str += textwrap.indent(self._callback._script.daphnedsl_script, prefix="    ")
        multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._callback.daphnedsl_name};\n", prefix="    ")
        multiline_str += textwrap.indent(f"{self._pred._script.daphnedsl_script}", prefix="    ")
        multiline_str += "}"
        self.daphnedsl_name = named_input_vars['node']
        # try
        
        
        return multiline_str

    def compute(self) -> None:
        return super().compute()