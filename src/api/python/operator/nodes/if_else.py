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
import textwrap

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

from api.python.utils import analyzer

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class IfElse(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', pred, true_fn, false_fn,
                 named_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'IfElse':
        _named_input_nodes = named_input_nodes

        outer_vars_true = analyzer.get_outer_scope_variables(true_fn)
        outer_vars_false = analyzer.get_outer_scope_variables(false_fn)

        outer_vars_both = {**outer_vars_true, **outer_vars_false}
        for i, var in enumerate(outer_vars_both):
            node = outer_vars_both[var]
            if node:
                _named_input_nodes.update({f"i_var{i}": node})
        
        outer_vars_pred = analyzer.get_outer_scope_variables(pred)

        for i, var in enumerate(outer_vars_pred):
            node = outer_vars_pred[var]
            if node:
                _named_input_nodes.update({f"pred_var{i}": node})

        _named_input_nodes.update({'cond': pred()})
        
        super().__init__(daphne_context, 'for_loop', named_input_nodes=named_input_nodes,
                         output_type=OutputType.MATRIX)

        self._true_fn = true_fn(named_input_nodes['node'])
        self._false_fn = false_fn(named_input_nodes['node'])
        
    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        #print(f"input vars: {named_input_vars}")
        parent_level = 1  # default
        if self._script:
            parent_level = self._script._nested_level

        self._true_fn._script = NestedDaphneDSLScript(self._true_fn.daphne_context, parent_level+1)
        self._true_fn._script.build_code(self._true_fn)
        self._false_fn._script = NestedDaphneDSLScript(self._false_fn.daphne_context, parent_level+1)
        self._false_fn._script.build_code(self._false_fn)
        
        pred_var = named_input_vars['cond']
        if pred_var in ["TRUE", "FALSE"]:
            pred_var = str(int(pred_var == "TRUE"))
        
        multiline_str = f"if ({pred_var}) {{\n"
        multiline_str += textwrap.indent(self._true_fn._script.daphnedsl_script, prefix="    ")
        multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._true_fn.daphnedsl_name};\n", prefix="    ")
        multiline_str += "}"
        multiline_str += " else {\n"
        multiline_str += textwrap.indent(self._false_fn._script.daphnedsl_script, prefix="    ")
        multiline_str += textwrap.indent(f"{named_input_vars['node']}={self._false_fn.daphnedsl_name};\n", prefix="    ")
        multiline_str += "}"
        self.daphnedsl_name = named_input_vars['node']
        return multiline_str

    def compute(self) -> None:
        return super().compute()