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
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'ForLoop':
        if analyzer.get_number_argument(callback) != (len(unnamed_input_nodes) + 1):
            raise ValueError(f"{callback} does not have the same number of arguments as input nodes + 1")

        _named_input_nodes = named_input_nodes
        self.iter_num = Scalar(self, "0", assign=True)
        _named_input_nodes.update({"iter": self.iter_num})

        self._callback = callback(*unnamed_input_nodes, _named_input_nodes['iter'])
        if (not isinstance(self._callback, tuple)):
            self._callback = (self._callback, )

        if len(self._callback) != len(unnamed_input_nodes):
            raise ValueError(f"{callback} and does not have the same number return values as input nodes")

        outer_vars = analyzer.get_outer_scope_variables(callback)
        for i, var in enumerate(outer_vars):
            inside_node = callback.__globals__.get(var)
            if var:
                _named_input_nodes.update({f"var{i}": inside_node})

        self.copy = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(self, "+", [node, 0])
            new_matrix_node._source_node = self
            self.copy.append(new_matrix_node)
        
        super().__init__(daphne_context, 'for_loop', unnamed_input_nodes=unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def get_copy(self):
        return tuple(self.copy)
        

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        parent_level = 1  # default
        if self._script:
            parent_level = self._script._nested_level

        script = NestedDaphneDSLScript(self.daphne_context, parent_level+1)
        names = script.build_code(self._callback)

        multiline_str = str()
        multiline_str += f"for({named_input_vars['iter']} in {named_input_vars['start']}:{named_input_vars['end']}:{named_input_vars['step']}) {{\n"
        multiline_str += textwrap.indent(script.daphnedsl_script, prefix="    ")
        for i, name in enumerate(names):
            multiline_str += textwrap.indent(f"{unnamed_input_vars[i]}={name};\n", prefix="    ")
        multiline_str += "}"

        return multiline_str

        # multiline_str += f"{self.daphnedsl_name} = 0;\n"
        # multiline_str += f"for({named_input_vars['iter']} in {named_input_vars['start']}:{named_input_vars['end']}:{named_input_vars['step']}) {{\n"
        # multiline_str += textwrap.indent(self._callback._script.daphnedsl_script, prefix="    ")
        # multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._callback.daphnedsl_name};\n", prefix="    ")
        # multiline_str += "}"
        # self.daphnedsl_name = named_input_vars['node']
        # return multiline_str

    def compute(self) -> None:
        raise NotImplementedError("'ForLoop' node is not intended to be computed")