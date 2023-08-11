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
