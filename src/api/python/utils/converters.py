from api.python.utils.consts import VALID_INPUT_TYPES
import numpy as np
from api.python.operator.nodes.matrix import Matrix
from typing import Sequence, Dict, Union


def from_numpy(mat: np.array,
                   *args: Sequence[VALID_INPUT_TYPES],
                   **kwargs: Dict[str, VALID_INPUT_TYPES]) -> Matrix:
        """Generate DAGNode representing matrix with data given by a numpy array, which will be sent to SystemDS
        on need.
        :param mat: the numpy array
        :param args: unnamed parameters
        :param kwargs: named parameters
        :return: A Matrix
        """

        unnamed_params = ['\'./tmp/{file_name}\'']

        if len(mat.shape) == 2:
            named_params = {'rows': mat.shape[0], 'cols': mat.shape[1]}
        elif len(mat.shape) == 1:
            named_params = {'rows': mat.shape[0], 'cols': 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params.update(kwargs)
        return Matrix( 'read', unnamed_params, named_params, local_data=mat)

def rand( rows: int, cols: int,
             min: Union[float, int] = None, max: Union[float, int] = None,sparsity: Union[float, int] = 0, seed: Union[float, int] = 0
             ) -> 'Matrix':
        """Generates a matrix filled with random values
        :param rows: number of rows
        :param cols: number of cols
        :param min: min value for cells
        :param max: max value for cells
        :param pdf: "uniform"/"normal"/"poison" distribution
        :param sparsity: fraction of non-zero cells
        :param seed: random seed
        :param lambd: lamda value for "poison" distribution
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