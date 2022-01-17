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
from api.python.operator.operation_node import OperationNode
from api.python.script_building.dag import OutputType
from api.python.utils.consts import VALID_INPUT_TYPES
import numpy as np
from api.python.operator.nodes.matrix import Matrix
from typing import Sequence, Dict, Union

#TODO: load data from numpy array
def from_numpy(mat: np.array,
                   *args: Sequence[VALID_INPUT_TYPES],
                   **kwargs: Dict[str, VALID_INPUT_TYPES]) -> Matrix:
        """Generate DAGNode representing matrix with data given by a numpy array.
        :param mat: the numpy array
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Matrix
        """


        unnamed_params = ['"src/api/python/tmp/{file_name}.csv\"']
        
       # if len(mat.shape) == 2:
          #  named_params = {'rows': mat.shape[0], 'cols': mat.shape[1], 'delim:':'","'}
       # elif len(mat.shape) == 1:
           # named_params = {'rows': mat.shape[0], 'cols': 1, 'delim:':'","'}
       # else:
            # TODO Support tensors.
        #    raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params = []
        #named_params.update(kwargs)
        return Matrix( 'readMatrix', unnamed_params, named_params, local_data=mat)


def rand( rows: int, cols: int,
             min: Union[float, int] = None, max: Union[float, int] = None,sparsity: Union[float, int] = 0, seed: Union[float, int] = 0
             ) -> 'Matrix':
        """Generates a matrix filled with random values
        :param rows: number of rows
        :param cols: number of cols
        :param min: min value for cells
        :param max: max value for cells
        :param sparsity: fraction of non-zero cells
        :param seed: random seed
        :return:
        """
        if rows < 0:
            raise ValueError("In rand statement, can only assign rows a long (integer) value >= 0 "
                             "-- attempted to assign value: {r}".format(r=rows))
        if cols < 0:
            raise ValueError("In rand statement, can only assign cols a long (integer) value >= 0 "
                             "-- attempted to assign value: {c}".format(c=cols))
        #num of rows, cols, min, max, sparsity, seed
        named_input_nodes = {
            'rows': rows, 'cols': cols, 'min': min, 'max':max, 'sparsity':sparsity, 'seed':seed}
        if min is not None:
            named_input_nodes['min'] = min
        if max is not None:
            named_input_nodes['max'] = max
        if sparsity is not None:
            named_input_nodes['sparsity'] = sparsity
        if seed is not None:
            named_input_nodes['seed'] = seed

        return Matrix('rand', [], named_input_nodes=named_input_nodes)