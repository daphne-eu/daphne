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
from api.python.operator.nodes.frame import Frame
from api.python.operator.nodes.scalar import Scalar
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
from api.python.script_building.nested_script import NestedDaphneDSLScript
from api.python.utils import analyzer

from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Callable
import textwrap

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext


class Cond(OperationNode):
    _outputs: Iterable['OperationNode']

    def __init__(self, daphne_context: 'DaphneContext', pred: Callable, then_fn: Callable, else_fn: Callable,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None) -> 'Cond':
        """
        Operational node that represents if-then-else statement functionality.
        Its reserved variable is left unused and is not added ot the generated script.

        :param daphne_context:
        :param pred: function with single scalar output representing the predicate
        :param then_fn: function for the logic of the TRUE-block
        :param else_fn: function for the logic of the FALSE-block
        :param unnamed_input_nodes: operation nodes that are up for manipulation
        """
        self.nested_level = 0  # default value
        _named_input_nodes = dict()
        # cast the iterable to list for consistency and to avoid additional copying
        _unnamed_input_nodes = list(unnamed_input_nodes)
        # analyze if the passed functions fulfill the requirements
        if else_fn is not None:
            if analyzer.get_number_argument(then_fn) != analyzer.get_number_argument(else_fn):
                raise ValueError(f"{then_fn} and {else_fn} do not have the same number of arguents")
            elif analyzer.get_number_argument(then_fn) != len(unnamed_input_nodes):
                raise ValueError(f"{then_fn} and {else_fn} do not have the same number of arguments as input nodes")
        else:
            if analyzer.get_number_argument(then_fn) != len(unnamed_input_nodes):
                raise ValueError(f"{then_fn} does not have the same number of arguments as input nodes")

        if analyzer.get_number_argument(pred) != 0:
            raise ValueError(f"{pred} has more then 0 arguments")
        elif isinstance(pred(), tuple):
            raise ValueError(f"{pred} has more then 1 return values")

        # spare storing the arguments additionally by redefined the functions
        self.then_fn = lambda: then_fn(*unnamed_input_nodes)
        self.else_fn = None
        if else_fn:
            self.else_fn = lambda: else_fn(*unnamed_input_nodes)
        # get the variables in outer scope to the according functions
        outer_vars_then = analyzer.get_outer_scope_variables(then_fn)
        outer_vars_else = analyzer.get_outer_scope_variables(else_fn) if else_fn else dict()
        outer_vars_both = {**outer_vars_then, **outer_vars_else}
        outer_vars_pred = analyzer.get_outer_scope_variables(pred)
        # append the outer scope variables to input nodes so these
        # can be defined upfront by the depth-first-search pass
        for node in outer_vars_both.values():
            if node:
                _unnamed_input_nodes.append(node)
        for node in outer_vars_pred.values():
            if node:
                _unnamed_input_nodes.append(node)
        # evaluate the predicate upfront
        _named_input_nodes.update({'pred': pred()})
        # TODO: decide if here is the best place for this piece of code: maybe just after the first analysis
        # initiate the output operation nodes
        self._outputs = list()
        for node in unnamed_input_nodes:
            new_matrix_node = None
            if isinstance(node, Matrix):
                new_matrix_node = Matrix(daphne_context, None, [node], copy=True)
            elif isinstance(node, Frame):
                new_matrix_node =  Frame(daphne_context, None, [node], copy=True)
            elif isinstance(node, Scalar):  
                new_matrix_node = Scalar(daphne_context, None, [node], copy=True)
            else:
                raise ValueError(f"Unsupported input node type {type(node)}")
            new_matrix_node._source_node = self
            self._outputs.append(new_matrix_node)
        
        super().__init__(daphne_context, 'cond', unnamed_input_nodes=_unnamed_input_nodes, named_input_nodes=_named_input_nodes,
                         output_type=OutputType.NONE)

    def __getitem__(self, index) -> Tuple['Matrix']:
        return self._outputs[index]
        
    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        """
        Generates the DaphneDSL code block for the if-then-else statement.
        Here the 'then_fn' and 'else_fn' (if not None) are being
        evaluated and then the code lines for their according functionalities
        are generated and added inside the if-then-else code structure.

        :param var_name: variable name reserved for the operation node - NOT used
        :param unnamed_input_vars:
        :param named_input_vars:
        :return:
        """
        # get tuple of output operation nodes for the 'then_fn'
        then_fn_outputs = self.then_fn()
        if not isinstance(then_fn_outputs, tuple):
            then_fn_outputs = (then_fn_outputs, )
        # generate the code lines for the 'then_fn' functionality
        then_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1)
        # get the inner scope variable names storing the output operation nodes
        then_names = then_script.build_code(then_fn_outputs)
        # store the generated code lines as string
        then_body = then_script.daphnedsl_script
        # assignment of the inner scope variable names to the variables of the outer scope
        for i, name in enumerate(then_names):
            then_body += f"{unnamed_input_vars[i]}={name};\n"
        # pack all code lines in the if-statement structure
        multiline_str = str()
        multiline_str += f"if ({named_input_vars['pred']}) {{\n"
        multiline_str += textwrap.indent(then_body, prefix="    ")
        multiline_str += "}"

        if self.else_fn:
            # get tuple of output operation nodes for the 'else_fn'
            else_fn_outputs = self.else_fn()
            if not isinstance(else_fn_outputs, tuple):
                else_fn_outputs = (else_fn_outputs,)
            # generate the code lines for the 'else_fn' functionality
            else_script = NestedDaphneDSLScript(self.daphne_context, self.nested_level + 1)
            # get the inner scope variable names storing the output operation nodes
            else_names = else_script.build_code(else_fn_outputs)
            # store the generated code lines as string
            else_body = else_script.daphnedsl_script
            # assignment of the inner scope variable names to the variables of the outer scope
            for i, name in enumerate(else_names):
                else_body += f"{unnamed_input_vars[i]}={name};\n"
            # pack all code lines in the else-statement structure
            multiline_str += " else {\n"
            multiline_str += textwrap.indent(else_body, prefix="    ")
            multiline_str += "}"

        return multiline_str 

    def compute(self) -> None:
        raise NotImplementedError("'Cond' node is not intended to be computed")