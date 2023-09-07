import inspect
from typing import Callable, get_type_hints
from api.python.operator.operation_node import OperationNode

def get_outer_scope_variables(function: Callable):
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
    return function.__code__.co_argcount

def get_argument_types(function: Callable):
    arguments = inspect.signature(function).parameters
    arguments_type_hints = get_type_hints(function)
    return {key: arguments_type_hints.get(key) for key in arguments.keys()}
