import inspect
import ast
from typing import Callable
import warnings

def get_outer_scope_variables(function: Callable):
    ast_tree = ast.parse(inspect.getsource(function))

    inner_scope_vars = set()
    outer_scope_vars = set()

    # iterate over the tree
    this_func_definition_parsed: bool = False
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef):
            if this_func_definition_parsed:
                warnings.warn(f"Nested function can lead to problems")
        elif isinstance(node, ast.If):
            warnings.warn(f"If-statements can lead to problems")
        elif isinstance(node, ast.For):
            warnings.warn(f"For-loops can lead to problems")
        elif isinstance(node, ast.While):
            warnings.warn(f"While-loops can lead to problems")
        elif isinstance(node, ast.Assign):
            # Analyze variable assignments
            targets = [target.id for target in node.targets]
            inner_scope_vars.update(targets)
        elif isinstance(node, ast.Call):
            # Analyze function calls
            if isinstance(node.func, ast.Name):
                inner_scope_vars.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                inner_scope_vars.add(node.func.value.id)
        elif isinstance(node, ast.Name):
            # Analyze variable names
            if (node.id not in inner_scope_vars):
                outer_scope_vars.add(node.id)
            else:
                inner_scope_vars.add(node.id)
        elif isinstance(node, ast.arg):
            #print("Argument: ", node.arg)
            inner_scope_vars.add(node.arg)

    return list(outer_scope_vars)