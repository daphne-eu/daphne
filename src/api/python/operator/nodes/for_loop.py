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

class ForLoop(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', callback,
                 named_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'ForLoop':
        _named_input_nodes = named_input_nodes

        outer_vars = analyzer.get_outer_scope_variables(callback)
        for i, var in enumerate(outer_vars):
            inside_node = callback.__globals__.get(var)
            if var:
                _named_input_nodes.update({f"var{i}": inside_node})
        
        self.iter_num = Scalar(self, "0", assign=True)
        _named_input_nodes.update({"iter": self.iter_num})
        
        print(daphne_context.__dict__)
        super().__init__(daphne_context, 'for_loop', named_input_nodes=named_input_nodes,
                         output_type=OutputType.MATRIX)
        print(f"outer scope vars: {analyzer.get_outer_scope_variables(callback)}")
        
        self._callback = callback(named_input_nodes['node'], named_input_nodes['iter'])
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        print(f"input vars: {named_input_vars}")
        parent_level = 1  # default
        if self._script:
            parent_level = self._script._nested_level

        self._callback._script = NestedDaphneDSLScript(self._callback.daphne_context, parent_level+1)
        self._callback._script.build_code(self._callback)
        #print(self._callback._script.daphnedsl_script)
        multiline_str = f"for({named_input_vars['iter']} in {named_input_vars['start']}:{named_input_vars['end']}:{named_input_vars['step']}) {{\n"
        #multiline_str += f"    {named_input_vars['number']} = {named_input_vars['node']} + {self.daphnedsl_name};
        multiline_str += textwrap.indent(self._callback._script.daphnedsl_script, prefix="    ")
        multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._callback.daphnedsl_name};\n", prefix="    ")
        multiline_str += "}"
        self.daphnedsl_name = named_input_vars['node']
        return multiline_str

    def compute(self) -> None:
        return super().compute()