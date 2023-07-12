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

# Overview: DAPHNE's Python API

DaphneLib is a simple user-facing Python API that allows calling individual basic and higher-level DAPHNE built-in functions.
The overall design follows similar abstractions like PySpark and Dask by using lazy evaluation. When the evaluation is triggered, DaphneLib assembles and executes a [DaphneDSL](/doc/DaphneDSL/LanguageRef.md) script that uses the entire DAPHNE compilation and runtime stack, including all optimizations.
Users can easily mix and match DAPHNE computations with other Python libraries and plotting functionality.

**DaphneLib is still in an experimental stage, feedback and bug reports via GitHub issues are highly welcome.**

## Introductory Example

The following simple example script generates a *5x3* matrix of random values in *[0, 1)* using numpy, imports the data to DAPHNE, and shifts and scales the data such that each column has a mean of *0* and a standard deviation of *1*.

```python
# (1) Import DaphneLib.
from api.python.context.daphne_context import DaphneContext
import numpy as np

# (2) Create DaphneContext.
dc = DaphneContext()

# (3) Obtain a DAPHNE matrix.
X = dc.from_numpy(np.random.rand(5, 3))

# (4) Define calculations in DAPHNE.
Y = (X - X.mean(axis=1)) / X.stddev(axis=1)

# (5) Compute result in DAPHNE.
print(Y.compute())
```

First, DAPHNE's Python library must be imported **(1)**.
We plan to make this as simple as `import daphne` in the future.

Then, a `DaphneContext` must be created **(2)**.
The `DaphneContext` offers means to obtain DAPHNE matrices and frames, which serve as the starting point for defining complex computations.
Here, we generate some random data in numpy and import it to DAPHNE **(3)**.

Based on DAPHNE matrices/frames/scalars (and Python scalars), complex expressions can be defined **(4)** using Python operators (such as `-` and `/` above) and methods on the DAPHNE matrices/frames/scalars (such as `mean()` and `stddev()` above).
The results of these expressions again represent DAPHNE matrices/frames/scalars.

Up until here, no acutal computations are performed.
Instead, an internal DAG (directed acyclic graph) representation of the computation is built.
When calling `compute()` on the result **(5)**, the DAG is automatically optimized and executed by DAPHNE.
This principle is known as *lazy evaluation*.
(Internally, a [DaphneDSL](/doc/DaphneDSL/LanguageRef.md) script is created, which is sent through the entire DAPHNE compiler and runtime stack, thereby profiting from all optimizations in DAPHNE.)
The result is returned as a `numpy.ndarray` (for DAPHNE matrices), as a `pandas.DataFrame` (for DAPHNE frames), or as a plain Python scalar (for DAPHNE scalars), and can then be further used in Python.

The script above can be executed by:

```bash
python3 scripts/examples/daphnelib/shift-and-scale.py
```

Note that there are some **temporary limitations** (which will be fixed in the future):

- `python3` must be executed from the DAPHNE base directory.
- Before executing DaphneLib Python scripts, the environment variable `PYTHONPATH` must be updated by executing the following command once per session:

  ```bash
  export PYTHONPATH="$PYTHONPATH:$PWD/src/"
  ```

The remainder of this document presents the core features of DaphneLib *as they are right now*, but *note that DaphneLib is still under active development*.

## Data and Value Types

DAPHNE differentiates *data types* and *value types*.

Currently, DAPHNE supports the following *abstract* **data types**:

- `matrix`: homogeneous value type for all cells
- `frame`: a table with columns of potentially different value types
- `scalar`: a single value

**Value types** specify the representation of individual values. We currently support:

- floating-point numbers of various widths: `f64`, `f32`
- signed and unsigned integers of various widths: `si64`, `si32`, `si8`, `ui64`, `ui32`, `ui8`
- strings `str` *(currently only for scalars, support for matrix elements is still experimental)*
- booleans `bool` *(currently only for scalars)*

Data types and value types can be combined, e.g.:

- `matrix<f64>` is a matrix of double-precision floating point values

In DaphneLib, each node of the computation DAG has one of the types `api.python.operator.nodes.matrix.Matrix`, `api.python.operator.nodes.frame.Frame`, or `api.python.operator.nodes.scalar.Scalar`.
The type of a node determines which methods can be invoked on it (see [DaphneLib API reference](/doc/DaphneLib/APIRef.md)).

## Obtaining DAPHNE Matrices and Frames

The `DaphneContext` offers means to obtain DAPHNE matrices and frames, which serve as the starting point for defining complex computations.
More precisely, DAPHNE matrices and frames can be obtained in the following ways:

- importing data from other Python libraries (e.g., numpy and pandas)
- generating data in DAPHNE (e.g., random data, constants, or sequences)
- reading files using DAPHNE's readers (e.g., CSV, Matrix Market, Parquet, DAPHNE binary format)

A comprehensive list can be found in the [DaphneLib API reference](/doc/DaphneLib/APIRef.md#daphnecontext).

## Building Complex Computations

Based on DAPHNE matrices/frames/scalars and Python scalars, complex expressions can be defined by

- Python operators
- DAPHNE matrix/frame/scalar methods

The results of these expressions again represent DAPHNE matrices/frames/scalars.

### Python Operators

DaphneLib currently supports the following binary operators on DAPHNE matrices/frames/scalars:

| Operator | Meaning |
| --- | --- |
| `@` | matrix multiplication |
| `*`, `/` | multiplication, division |
| `+`, `-` | addition/string concatenation, subtraction |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | comparison |

*We plan to add more operators, including unary operators.*

*Matrix multiplication (`@`):*
The inputs must be matrices of compatible shapes, and the output is always a matrix.

*All other operators:*
The following table shows which combinations of inputs are allowed and which result they yield:

| Left input | Right input | Result | Details |
| --- | --- | --- | --- |
| scalar | scalar | scalar | binary operation of two scalars |
| matrix (n x m) | scalar | matrix (n x m) | element-wise operation of each value with scalar |
| scalar | matrix (n x m) | matrix (n x m) | element-wise operation of scalar with each value **(*)** |
| matrix (n x m) | matrix (n x m) | matrix (n x m) | element-wise operation on corresponding values |
| matrix (n x m) | matrix (1 x m) | matrix (n x m) | broadcasting of row-vector |
| matrix (n x m) | matrix (n x 1) | matrix (n x m) | broadcasting of column-vector |

**(\*)** *Scalar-`op`-matrix* operations are so far only supported for `+`, `-`, `*`, `/`; for `/` only if the matrix is of a floating-point value type.

In the future, we will fully support *scalar-`op`-matrix* operations as well as row/column-matrices as the left-hand-side operands.

*Examples:*

```r
1.5 * X @ y + 0.001
```

### Matrix/Frame/Scalar Methods

DaphneLib's classes `Matrix`, `Frame`, and `Scalar` offer a range of methods to call DAPHNE built-in functions.
A comprehensive list can be found in the [DaphneLib API reference](/doc/DaphneLib/APIRef.md#building-complex-computations).

*Examples:*

```r
X.t()
X.sqrt()
X.cbind(Y)
```

## Data Exchange with other Python Libraries

DaphneLib will support efficient data exchange with other well-known Python libraries, in both directions.
The data transfer from other Python libraries to DaphneLib can be triggered through the `from_...()` methods of the `DaphneContext` (e.g., `from_numpy()`).
A comprehensive list of these methods can be found in the [DaphneLib API reference](/doc/DaphneLib/APIRef.md#daphnecontext).
The data transfer from DaphneLib back to Python happens during the call to `compute()`.
If the result of the computation in DAPHNE is a matrix, `compute()` returns a `numpy.ndarray`; if the result is a frame, it returns a `pandas.DataFrame`; and if the result is a scalar, it returns a plain Python scalar.

So far, DaphneLib can exchange data with numpy (via shared memory) and  pandas (via CSV files).
Enabling data exchange with TensorFlow and PyTorch is on our agenda.
Furthermore, we are working on making the data exchange more efficient in general.

### Data Exchange with numpy

*Example:*

```python
from api.python.context.daphne_context import DaphneContext
import numpy as np

dc = DaphneContext()

# Create data in numpy.
a = np.arange(8.0).reshape((2, 4))

# Transfer data to DaphneLib (lazily evaluated).
X = dc.from_numpy(a)

print("How DAPHNE sees the data from numpy:")
X.print().compute()

# Add 100 to each value in X.
X = X + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100 to each value, back in Python:")
print(X.compute())
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-numpy.py
```

*Output:*

```text
How DAPHNE sees the data from numpy:
DenseMatrix(2x4, double)
0 1 2 3
4 5 6 7

Result of adding 100 to each value, back in Python:
[[100. 101. 102. 103.]
 [104. 105. 106. 107.]]
```

### Data Exchange with pandas

*Example:*

```python
from api.python.context.daphne_context import DaphneContext
import pandas as pd

dc = DaphneContext()

# Create data in pandas.
df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, -2.2, 3.3]})

# Transfer data to DaphneLib (lazily evaluated).
F = dc.from_pandas(df)

print("How DAPHNE sees the data from pandas:")
F.print().compute()

# Append F to itself.
F = F.rbind(F)

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of appending the frame to itself, back in Python:")
print(F.compute())
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-pandas.py
```

*Output:*

```text
How DAPHNE sees the data from pandas:
Frame(3x2, [a:int64_t, b:double])
1 1.1
2 -2.2
3 3.3

Result of appending the frame to itself, back in Python:
   a    b
0  2 -2.2
1  3  3.3
2  1  1.1
3  2 -2.2
4  3  3.3
```

## Known Limitations

DaphneLib is still in an early development stage.
Thus, there are a few limitations that users should be aware of.
We plan to fix all of these limitations in the future.

- `import`ing DaphneLib is still unnecessarily verbose.
- Using DAPHNE's command-line arguments to influence its behavior is not supported yet.
- Many DaphneDSL built-in functions are not represented by DaphneLib methods yet.
- Complex control flow (if-then-else, loops, functions) are not supported yet. Python control flow statements are of limited applicability for DaphneLib.
- High-level primitives for integrated data analysis pipelines, which are implemented in DaphneDSL, cannot be called from DaphneLib yet.
