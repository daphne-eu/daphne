<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# API Reference

This document is a hand-crafted reference of the DaphneLib API.
A general introduction to [DaphneLib (DAPHNE's Python API)](/doc/DaphneLib/Overview.md) can be found in a separate document.
DaphneLib will offer numerous methods for *obtaining DAPHNE matrices and frames* as well as for *building complex computations* based on them.
Ultimately, DaphneLib will support all [DaphneDSL built-in functions](/doc/DaphneDSL/Builtins.md) on matrices and frames.
Futhermore, **we also plan to create a library of higher-level primitives** allowing users to productively implement integrated data analysis pipelines at a much higher level of abstraction.

At the moment, the documentation is still rather incomplete.
However, as the methods largely map to DaphneDSL built-in functions, you can find some more information in the [List of DaphneDSL built-in functions](/doc/DaphneDSL/Builtins.md), for the time being.

## Obtaining DAPHNE Matrices and Frames

### `DaphneContext`

**Importing data from other Python libraries:**

- **`from_numpy`**`(mat: np.array, shared_memory=True) -> Matrix`
- **`from_pandas`**`(df: pd.DataFrame) -> Frame`
  
**Generating data in DAPHNE:**

- **`fill`**`(arg, rows:int, cols:int) -> Matrix`
- **`seq`**`(start, end, inc) -> Matrix`
- **`rand`**`(rows: int, cols: int, min: Union[float, int] = None, max: Union[float, int] = None, sparsity: Union[float, int] = 0, seed: Union[float, int] = 0) -> Matrix`

**Reading files using DAPHNE's readers:**

- **`readMatrix`**`(file:str) -> Matrix`
- **`readFrame`**`(file:str) -> Frame`

## Building Complex Computations

Complex computations can be built using Python operators (see [DaphneLib](/doc/DaphneLib/Overview.md)) and using DAPHNE matrix/frame/scalar methods.
In the following, we describe only the latter.

### `Matrix` API Reference

**Data Generation:**

- **`diagMatrix`**`()`

**Matrix dimensions:**

- **`ncol`**`()`
- **`nrow`**`()`

**Elementwise unary:**

- **`sqrt`**`()`

**Elementwise binary:**

- **`max`**`(other: 'Matrix')`
- **`min`**`(other: 'Matrix')`

**Aggregation:**

- **`sum`**`(axis: int = None)`
- **`aggMin`**`(axis: int = None)`
- **`aggMax`**`(axis: int = None)`
- **`mean`**`(axis: int = None)`
- **`stddev`**`(axis: int = None)`

**Reorganization:**

- **`t`**`()`

**Other matrix operations:**

- **`solve`**`(other: 'Matrix')`

**Input/output:**

- **`print`**`()`
- **`write`**`(file: str)`

### `Frame` API Reference

**Frame dimensions:**

- **`nrow`**`()`
- **`ncol`**`()`

**Reorganization:**

- **`cbind`**`(other)`
- **`rbind`**`(other)`

**Extended relational algebra:**

- **`cartesian`**`(other)`

**Input/output:**

- **`print`**`()`
- **`write`**`(file: str)`

### `Scalar` API Reference

**Unary operations:**

- **`sqrt`**`()`

**Input/output:**

- **`print`**`()`


## Building Complex Control Structures
Complex control structures like `if-then-else`, `for-loops`, `while-loops` and `do-while-loops` can be built using DAPHNE context methods. These can be used to manipulate matrices only (DAPHNE matrix) and are lazy evaluated. Futhermore, user-defined functions can created to build more complex code snippets which can then be again lazy evaluated. The user-defined functions can manipulate all DAPHNE data objects (matrix/frame/scalar).

The following references assume that the required DAPHNE context object is initialized as:
```python
dctx = DaphneContext()
```
\* `VALID_CUMPUTED_TYPES=Union['Matrix', 'Frame', 'Scalar']`

### **if-then-else**
**`dctx.cond(input_nodes, pred, true_fn, false_fn)`**

* input_nodes: Iterable[VALID_CUMPUTED_TYPES]
* pred: Callable  *(0 arguments, 1 return value)*
* true_fn: Callable  *(n arguments, n return values, n=[1, ...])*
* false_fn: Callable  *(n arguments, n return values, n=[1, ...])*
* returns: Tuple[VALID_CUMPUTED_TYPES]  *(length n)*

### **for-loops**
**`dctx.for_loop(input_nodes, callback, start, end, step)`**

* input_nodes: Iterable[VALID_CUMPUTED_TYPES]
* callback: Callable  *(n+1 arguments, n return values, n=[1, ...])*
* start: int
* end: int
* step: Union[int, None]
* returns: Tuple[VALID_CUMPUTED_TYPES]  *(length n)*

\* *callback* expects as last argument the interation variable and this is suppose to be used as a scalar.

### **while-loops**
**`dctx.while_loop(input_nodes, cond, callback)`**

* input_nodes: Iterable[VALID_CUMPUTED_TYPES]
* cond: Callable  *(n arguments, 1 return value, n=[1, ...])*
* callback: Callable  *(n arguments, n return values)*
* returns: Tuple[VALID_CUMPUTED_TYPES]  *(length n)*

### **do-while-loops**
**`dctx.d0_while_loop(input_nodes, cond, callback)`**

* input_nodes: Iterable[VALID_CUMPUTED_TYPES]
* cond: Callable  *(n arguments, 1 return value, n=[1, ...])*
* callback: Callable  *(n arguments, n return values)*
* returns: Tuple[VALID_CUMPUTED_TYPES]  *(length n)*

### **user-defined functions**
**`@dctx.function`** <-> **`dctx.function(callback)`**

* callback: Callable
    - This function requires adding typing hints in case the arguments are suppossed 
    to be handled as `Scalar` or `Frame`, all arguments without hints are handled as
    `Matrix` objects. Hinting `Matrix` is optional. Wrong or missing typing hints can 
    trigger errors before and during computing (lazy evaluation).
* returns: Tuple[VALID_CUMPUTED_TYPE]  *(length equals the return values of callback)*

\* if the decorator is used the *callback* is defined right below it like regular Python method

### logical operators
Logical *and* (`&&`) and *or* (`||`) operators can be used for the conditions for `while-loops` and `do-while-loops` as well as for the predicates for `if-then-else` statements.

**`dctx.logical_and(left_operand, right_operand)`**

* left_operand: 'Scalar'
* right_operand: 'Scalar'
* returns: 'Scalar'

**`dctx.logical_or(left_operand, right_operand)`**

* left_operand: 'Scalar'
* right_operand: 'Scalar'
* returns: 'Scalar'
