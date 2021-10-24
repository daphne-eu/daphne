from typing import Union

VALID_INPUT_TYPES = Union['DAGNode', str, int, float, bool]
BINARY_OPERATIONS = ['+','-', '/', '//','*','<','<=','>', '>=', '==','!=', '%*%']
VALID_ARITHMETIC_TYPES = Union['DAGNode', int, float]