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

from api.python.script_building.dag import DAGNode, OutputType
from api.python.utils.consts import VALID_INPUT_TYPES, TMP_PATH, PROTOTYPE_PATH
from api.python.utils.daphnelib import DaphneLib

import ctypes
import os
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from context.daphne_context import DaphneContext

class DaphneDSLScript:
    daphnedsl_script :str
    inputs: Dict[str, DAGNode]
    out_var_name:List[str]
    _variable_counter: int

    def __init__(self, context) -> None:
        self.daphne_context = context
        self.daphnedsl_script = ''
        self.inputs = {}
        self.out_var_name = []
        self._variable_counter = 0
    
    def build_code(self, dag_root: DAGNode, type="shared memory"):
        baseOutVarString = self._dfs_dag_nodes(dag_root)
        if dag_root._output_type != OutputType.NONE:
            self.out_var_name.append(baseOutVarString)
            if dag_root.output_type == OutputType.MATRIX:
                if type == "files":
                    self.add_code(f'writeMatrix({baseOutVarString},"{TMP_PATH}/{baseOutVarString}.csv");')
                    return TMP_PATH +"/" + baseOutVarString + ".csv"
                elif type == "shared memory":
                    self.add_code(f'saveDaphneLibResult({baseOutVarString});')
                    return None
                else:
                    raise RuntimeError(f"unknown way to transfer the data: '{type}'")
            elif dag_root.output_type == OutputType.FRAME:
                self.add_code(f'writeFrame({baseOutVarString},"{TMP_PATH}/{baseOutVarString}.csv");')
                return TMP_PATH + "/" + baseOutVarString + ".csv"
            elif dag_root.output_type == OutputType.SCALAR:
                # We transfer scalars back to Python by wrapping them into a 1x1 matrix.
                self.add_code(f'saveDaphneLibResult(as.matrix({baseOutVarString}));')
                return None
            else:
                self.add_code(f'print({baseOutVarString});')
                return None
            

    def add_code(self, code:str)->None:
        """Add a line of DaphneDSL code to our script
        
        :param code: the DaphneDSL code line
        """
        self.daphnedsl_script += code +'\n'
    
    def clear(self, dag_root:DAGNode):
        self._dfs_clear_dag_nodes(dag_root)
        self._variable_counter = 0

    def execute(self):
        temp_out_file = open("tmpdaphne.daphne", "w")
        temp_out_file.writelines(self.daphnedsl_script)
        temp_out_file.close()
        
        #os.environ['OPENBLAS_NUM_THREADS'] = '1'
        res = DaphneLib.daphne(ctypes.c_char_p(b"tmpdaphne.daphne"))
        #os.environ['OPENBLAS_NUM_THREADS'] = '32'

    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES)->str:
        """Uses Depth-First-Search to create code from DAG
        :param dag_node: current DAG node
        :return: the variable name the current DAG node operation created
        """
        if not isinstance(dag_node, DAGNode):
            if isinstance(dag_node, bool):
                return 'true' if dag_node else 'false'
            return str(dag_node)

        # If the node already has a name, then it is already defined
        # in the script, therefore reuse.
        if dag_node.daphnedsl_name != "":
            return dag_node.daphnedsl_name
        
        if dag_node._source_node is not None:
            self._dfs_dag_nodes(dag_node._source_node)
        # For each node do the dfs operation and save the variable names in `input_var_names`.
        # Get variable names of unnamed parameters.
        unnamed_input_vars = [self._dfs_dag_nodes(input_node) for input_node in dag_node.unnamed_input_nodes]

        named_input_vars = {}
        if dag_node.named_input_nodes:
            for name, input_node in dag_node.named_input_nodes.items():
                named_input_vars[name] = self._dfs_dag_nodes(input_node)
                
        # Check if the node gets a name after multi-returns.
        # If it has, return that name.
        if dag_node.daphnedsl_name != "":
            return dag_node.daphnedsl_name

        dag_node.daphnedsl_name = self._next_unique_var()

        if dag_node.is_python_local_data:
            self.add_input_from_python(dag_node.daphnedsl_name, dag_node)

        code_line = dag_node.code_line(
            dag_node.daphnedsl_name, unnamed_input_vars, named_input_vars)
        self.add_code(code_line)
        return dag_node.daphnedsl_name

    def add_input_from_python(self, var_name: str, input_var: DAGNode) -> None:
        """Add an input for our preparedScript. Should only be executed for data that is python local.
        :param var_name: name of variable
        :param input_var: the DAGNode object which has data
        """
        self.inputs[var_name] = input_var

    def _dfs_clear_dag_nodes(self, dag_node:VALID_INPUT_TYPES)->str:
        if not isinstance(dag_node, DAGNode):
            return
        if not dag_node._daphnedsl_name:
            return
        dag_node._daphnedsl_name = ""
        for n in dag_node.unnamed_input_nodes:
            self._dfs_clear_dag_nodes(n)
        if dag_node._source_node is not None:
            self._dfs_clear_dag_nodes(dag_node._source_node)
            if dag_node._source_node.output_type == OutputType.MULTI_RETURN:
                for node in dag_node._source_node:
                    node._daphnedsl_name = ""
        if not dag_node.named_input_nodes:
            return
        for name,n in dag_node._named_input_nodes.items():
            self._dfs_clear_dag_nodes(n)

    def _next_unique_var(self)->str:
        var_id = self._variable_counter
        self._variable_counter += 1
        return f'V{var_id}'