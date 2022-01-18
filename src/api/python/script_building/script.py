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
# -------------------------------------------------------------
import os
from typing import List, Dict
from api.python.script_building.dag import DAGNode, OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
import numpy as np

class DSLScript:
    dsl_script :str
    inputs: Dict[str, DAGNode]
    out_var_name:List[str]
    _variable_counter: int

    def __init__(self) -> None:
        self.dsl_script = ''
        self.inputs = {}
        self.out_var_name = []
        self._variable_counter = 0
    
    def build_code(self, dag_root: DAGNode):
        baseOutVarString = self._dfs_dag_nodes(dag_root)
        if dag_root._output_type != OutputType.NONE:
            self.out_var_name.append(baseOutVarString)
            self.add_code(f'writeMatrix({baseOutVarString},"src/api/python/tmp/{baseOutVarString}.csv");')
            if("test/api/python" in os.getcwd()):
                path = "../../../"
            else:
                path = ""
            return np.genfromtxt(path+"src/api/python/tmp/"+baseOutVarString+".csv", delimiter=',')

    def add_code(self, code:str)->None:
        """Add a line of DSL code to our script
        
        :param code: the dsl code line
        """
        self.dsl_script += code +'\n'
    
    def clear(self, dag_root:DAGNode):
        self._dfs_clear_dag_nodes(dag_root)
        self._variable_counter = 0

    def execute(self):
        #todo: write all temporary files in /tmp
        #if not os.path.exists("../../../src/api/python/tmp/"):
         # os.mkdir("../../../src/api/python/tmp/")
        if(os.getcwd() != "/home/dzc/prototype"):
            temp_out_file = open("../../../tmpdaphne.daphne", "w")
            temp_out_file.writelines(self.dsl_script)
            temp_out_file.close()
            os.chdir("/home/dzc/prototype")
        else:
            temp_out_file = open("tmpdaphne.daphne", "w")
            temp_out_file.writelines(self.dsl_script)
            temp_out_file.close()
        os.system("build/bin/daphnec tmpdaphne.daphne")

    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES)->str:
        """depth first search to create code from DAG
        :param dag_node: current DAG node
        :return: variable name the current DAG node operation created
        """
        if not isinstance(dag_node, DAGNode):
            if isinstance(dag_node, bool):
                return 'TRUE' if dag_node else 'FALSE'
            return str(dag_node)
        #if node has name -> its already defined in the script -> reuse it
        if dag_node.dsl_name != "":
            return dag_node.dsl_name
        
        if dag_node._source_node is not None:
            self._dfs_dag_nodes(dag_node._source_node)
        
        unnamed_input_vars = [self._dfs_dag_nodes(input_node) for input_node in dag_node._unnamed_input_nodes]
        
        named_input_vars = {}
        if isinstance(dag_node.named_input_nodes, dict):
            for name, input_node in dag_node.named_input_nodes.items():
                
                named_input_vars[name] = self._dfs_dag_nodes(input_node)
                if isinstance(input_node, DAGNode) and input_node._output_type == OutputType.LIST:
                    dag_node._dsl_name = named_input_vars[name] + name
                    return dag_node._dsl_name

        
        if dag_node._dsl_name != "":
            return dag_node._dsl_name

        dag_node._dsl_name = self._next_unique_var()
        code_line = dag_node.code_line(dag_node.dsl_name, unnamed_input_vars, named_input_vars)
        self.add_code(code_line)
        return dag_node._dsl_name

    def _dfs_clear_dag_nodes(self, dag_node:VALID_INPUT_TYPES)->str:
        if not isinstance(dag_node, DAGNode):
            return
        dag_node._dsl_name = ""
        for n in dag_node.unnamed_input_nodes:
            self._dfs_clear_dag_nodes(n)
        if(isinstance(dag_node.named_input_nodes, dict)):
            for name,n in dag_node.named_input_nodes.items():
                self._dfs_clear_dag_nodes(n)
        if dag_node._source_node is not None:
            self._dfs_clear_dag_nodes(dag_node._source_node)

    def _next_unique_var(self)->str:
        var_id = self._variable_counter
        self._variable_counter += 1
        return f'V{var_id}'