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
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap
from copy import copy

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext


class Cond(OperationNode):
    def __init__(self, daphne_context: 'DaphneContext', pred: Callable, true_fn: Callable, false_fn: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'Cond':
        """
        Operational node that represents if-else statement functionality.
        Its reserved variable is left unused and is not added ot the generated script.

        :param daphne_context:
        :param pred: function with single scalar output representing the predicate
        :param true_fn: function for the logic of the TRUE-block
        :param false_fn: function for the logic of the FALSE-block
        :param unnamed_input_nodes: operation nodes that are up for manipulation
        """
        self.nested_level = 0  # default value
        _named_input_nodes = dict()
        _unnamed_input_nodes = copy(unnamed_input_nodes)
        # analyze if the passed functions fulfill the requirements
        if false_fn is not None:
            if analyzer.get_number_argument(true_fn) != analyzer.get_number_argument(false_fn):
                raise ValueError(f"{true_fn} and {false_fn} do not have the same number of arguents")
            elif analyzer.get_number_argument(true_fn) != len(unnamed_input_nodes):
                raise ValueError(f"{true_fn} and {false_fn} do not have the same number of arguments as input nodes")
        else:
            if analyzer.get_number_argument(true_fn) != len(unnamed_input_nodes):
                raise ValueError(f"{true_fn} does not have the same number of arguments as input nodes")

        if analyzer.get_number_argument(pred) != 0:
            raise ValueError(f"{pred} has more then 0 arguments")
        elif isinstance(pred(), tuple):
            raise ValueError(f"{pred} has more then 1 return values")

        # spare storing the arguments additionally by redefined the functions
        self.true_fn = lambda: true_fn(*unnamed_input_nodes)
        self.false_fn = None
        if false_fn:
            self.false_fn = lambda: false_fn(*unnamed_input_nodes)
        # get the variables in outer scope to the according functions
        outer_vars_true = analyzer.get_outer_scope_variables(true_fn)
        outer_vars_false = analyzer.get_outer_scope_variables(false_fn) if false_fn else dict()
        outer_vars_both = {**outer_vars_true, **outer_vars_false}
        outer_vars_pred = analyzer.get_outer_scope_variables(pred)
        # append the outer scope variables to inout nodes so these
        # can be defined upfront by the Deep-First-Search pass
        for node in outer_vars_both.values():
            if node:
                _unnamed_input_nodes.append(node)
        for node in outer_vars_pred.values():
            if node:
                _unnamed_input_nodes.append(node)
        # evaluate the predicate upfront
        _named_input_nodes.update({'pred': pred()})
        # ToDo: decide if here is the best place for this piece of code: maybe just after the fist analysis
        # initiate the output operation nodes
        self._output = list()
        for node in unnamed_input_nodes:
            new_matrix_node = Matrix(daphne_context, None, [node], copy=True)
            new_matrix_node._source_node = self
            self._output.append(new_matrix_node)
        
        super().__init__(daphne_context, 'cond', unnamed_input_nodes=_unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def get_output(self) -> Tuple['Matrix']:
        """
        :return: output operation nodes
        """
        return tuple(self._output)
        
    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        """
        Generates the DaphneDSL code block for if-else statement.
        Here the 'true_fn' and 'false_fn' (if not None) are being
        evaluated and then the code lines for their according functionalities
        are generated and added inside the if-else code structure.

        :param var_name: variable name reserved for the operation node - NOT used
        :param unnamed_input_vars:
        :param named_input_vars:
        :return:
        """
        # get tuple of output operation nodes for the 'true_fn'
        true_fn_outputs = self.true_fn()
        if not isinstance(true_fn_outputs, tuple):
            true_fn_outputs = (true_fn_outputs, )
        # generate the code lines for the 'true_fn' functionality
        true_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1)
        # get the inner scope variable names storing the output operation nodes
        true_names = true_script.build_code(true_fn_outputs)
        # store the generated code lines as string
        true_body = true_script.daphnedsl_script
        # assigment the inner scope variable names to the variables of the outer scope
        for i, name in enumerate(true_names):
            true_body += f"{unnamed_input_vars[i]}={name};\n"
        # pack all code lines in the if-statement structure
        multiline_str = str()
        multiline_str += f"if ({named_input_vars['pred']}) {{\n"
        multiline_str += textwrap.indent(true_body, prefix="    ")
        multiline_str += "}"

        if self.false_fn:
            # get tuple of output operation nodes for the 'false_fn'
            false_fn_outputs = self.false_fn()
            if not isinstance(false_fn_outputs, tuple):
                false_fn_outputs = (false_fn_outputs,)
            # generate the code lines for the 'false_fn' functionality
            false_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1)
            # get the inner scope variable names storing the output operation nodes
            false_names = false_script.build_code(false_fn_outputs)
            # store the generated code lines as string
            false_body = false_script.daphnedsl_script
            # assigment the inner scope variable names to the variables of the outer scope
            for i, name in enumerate(false_names):
                false_body += f"{unnamed_input_vars[i]}={name};\n"
            # pack all code lines in the else-statement structure
            multiline_str += " else {\n"
            multiline_str += textwrap.indent(false_body, prefix="    ")
            multiline_str += "}"

        return multiline_str 

    def compute(self) -> None:
        raise NotImplementedError("'Cond' node is not intended to be computed")