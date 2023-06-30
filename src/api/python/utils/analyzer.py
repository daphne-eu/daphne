import inspect
from typing import Callable
from api.python.operator.operation_node import OperationNode

def get_outer_scope_variables(function: Callable):
    # ToDo: Consider if free_vars should also be check to be isntace of OperationNode
    free_vars = inspect.getclosurevars(function).nonlocals
    globals = inspect.getclosurevars(function).globals
    key_to_delete = list()
    for name, obj in globals.items():
        if not isinstance(obj, OperationNode):
            key_to_delete.append(name)
    for key in key_to_delete:
        del globals[key]

    return {**free_vars, **globals}

def get_number_argument(function: Callable):
    arguments = inspect.signature(function).parameters
    return len(arguments)
