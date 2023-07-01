# DaphneLib Complex Control Flow
This documentation describe (for now) the functions for defining complex control flow in DaphneLib. 
The following descriptions assume that the required context object is defined as:
```python
dctx = DaphneContext()
```

## if-else statements
*dctx.cond(input_nodes, pred, true_fn, false_fn)*

* input_nodes: List['Matrix']
* pred: Callable  *(0 arguments, 1 return value)*
* true_fn: Callable  *(n arguments, n return values, n=[1, ...])*
* false_fn: Callable  *(n arguments, n return values, n=[1, ...])*
* returns: Tuple['Matrix']  *(length n)*

## for loops
*dctx.for_loop(input_nodes, callback, start, end, step)*

* input_nodes: List['Matrix']
* callback: Callable  *(n+1 arguments, n return values, n=[1, ...])*
* start: int
* end: int
* step: Union[int, None]
* returns: Tuple['Matrix']  *(length n)*

\* *callback* expects as last argument the interation variable and is to be used as scalar.

## while loops
*dctx.while_loop(input_nodes, cond, callback)*

* input_nodes: List['Matrix']
* cond: Callable  *(n arguments, 1 return value, n=[1, ...])*
* callback: Callable  *(n arguments, n return values)*
* returns: Tuple['Matrix']  *(length n)*

## user-defined functions
*@dctx.function* <-> *dctx.function(callback)*

* callback: Callable
* returns: Tuple['OperationNode']  *(length equals the return values of callback)*

## logical operators
### and-operator (`&&`)
*dctx.logacal_and(left_operand, right_operand)*

* left_operand: 'Scalar'
* right_operand: 'Scalar'
* returns: 'Scalar'

### or-operator (`||`)
*dctx.logacal_or(left_operand, right_operand)*

* left_operand: 'Scalar'
* right_operand: 'Scalar'
* returns: 'Scalar'




