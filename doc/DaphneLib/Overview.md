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

## Prerequisites

**Provide DAPHNE:**

- `libdaphnelib.so` and `libAllKernels.so` must be present
  - Building the project with `./build.sh --target daphnelib` achieves this (this creates a `lib` dir in the `daphne` project root)
  - OR use the `lib/` dir of a release
- `LD_LIBRARY_PATH` must be set (e.g., executed from `daphne/`: `export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH`)
- Set the environment variable `DAPHNELIB_DIR_PATH` to the path were the libraries (`*.so` files) are placed, e.g., `path/to/daphne/lib/`

## Installation

- There are two options to install the Python package `daphne` (DaphneLib)
  - Via github url: `pip install git+https://github.com/daphne-eu/daphne.git@main#subdirectory=src/api/python`
  - OR clone the DAPHNE repository and install from source files: `pip install daphne/src/api/python`
- *Recommendation:* Use a virtual environment

    ```shell
    python3 -m venv my_venv
    source my_venv/bin/activate
    pip install ...
    ```

## Use Without Installation

- In a cloned DAPHNE repository, DaphneLib can also be used without installing the Python package `daphne`
- To this end, `python3` must be told where to find the package by adding the respective directory to the Python path
- From the DAPHNE root directory, execute: `export PYTHONPATH="$PYTHONPATH:$PWD/src/api/python/"`
- OR execute the script `run_python.sh` (instead of `python3`) from the DAPHNE root directory, e.g., `./run_python.sh myScript.py`

## Introductory Example

The following simple example script generates a *5x3* matrix of random values in $[0, 1)$ using numpy, imports the data to DAPHNE, and shifts and scales the data such that each column has a mean of *0* and a standard deviation of *1*.

```python
# (1) Import DaphneLib.
from daphne.context.daphne_context import DaphneContext
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

In DaphneLib, each node of the computation DAG has one of the types `daphne.operator.nodes.matrix.Matrix`, `daphne.operator.nodes.frame.Frame`, or `daphne.operator.nodes.scalar.Scalar`.
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
| `**` | exponentiation |
| `*`, `/`, `%` | multiplication, division, modulo |
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

DaphneLib supports efficient data exchange with other well-known Python libraries, in both directions.
The data transfer from other Python libraries to DaphneLib can be triggered through the `from_...()` methods of the `DaphneContext` (e.g., `from_numpy()`).
A comprehensive list of these methods can be found in the [DaphneLib API reference](/doc/DaphneLib/APIRef.md#daphnecontext).
The data transfer from DaphneLib back to Python happens during the call to `compute()`.
If the result of the computation in DAPHNE is a matrix, `compute()` returns a `numpy.ndarray` (or optionally a `tensorflow.Tensor` or `torch.Tensor`); if the result is a frame, it returns a `pandas.DataFrame`; and if the result is a scalar, it returns a plain Python scalar.

So far, DaphneLib can exchange data with numpy, pandas, TensorFlow, and PyTorch.
By default, the data transfer is via shared memory (and in many cases zero-copy).

### Data Exchange with numpy

*Example:*

```python
from daphne.context.daphne_context import DaphneContext
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
from daphne.context.daphne_context import DaphneContext
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

### Data Exchange with TensorFlow

*Example:*

```python
from daphne.context.daphne_context import DaphneContext
import tensorflow as tf
import numpy as np

dc = DaphneContext()

print("========== 2D TENSOR EXAMPLE ==========\n")

# Create data in TensorFlow/numpy.
t2d = tf.constant(np.random.random(size=(2, 4)))

print("Original 2d tensor in TensorFlow:")
print(t2d)

# Transfer data to DaphneLib (lazily evaluated).
T2D = dc.from_tensorflow(t2d)

print("\nHow DAPHNE sees the 2d tensor from TensorFlow:")
T2D.print().compute()

# Add 100 to each value in T2D.
T2D = T2D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T2D.compute(asTensorFlow=True))

print("\n========== 3D TENSOR EXAMPLE ==========\n")

# Create data in TensorFlow/numpy.
t3d = tf.constant(np.random.random(size=(2, 2, 2)))

print("Original 3d tensor in TensorFlow:")
print(t3d)

# Transfer data to DaphneLib (lazily evaluated).
T3D, T3D_shape = dc.from_tensorflow(t3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from TensorFlow:")
T3D.print().compute()

# Add 100 to each value in T3D.
T3D = T3D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T3D.compute(asTensorFlow=True))
print("\nResult of adding 100, back in Python (with original shape):")
print(T3D.compute(asTensorFlow=True, shape=T3D_shape))
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-tensorflow.py
```

*Output (random numbers may vary):*

```text
========== 2D TENSOR EXAMPLE ==========

Original 2d tensor in TensorFlow:
tf.Tensor(
[[0.09682179 0.09636572 0.78658016 0.68227129]
 [0.64356184 0.96337785 0.07931763 0.97951051]], shape=(2, 4), dtype=float64)

How DAPHNE sees the 2d tensor from TensorFlow:
DenseMatrix(2x4, double)
0.0968218 0.0963657 0.78658 0.682271
0.643562 0.963378 0.0793176 0.979511

Result of adding 100, back in Python:
tf.Tensor(
[[100.09682179 100.09636572 100.78658016 100.68227129]
 [100.64356184 100.96337785 100.07931763 100.97951051]], shape=(2, 4), dtype=float64)

========== 3D TENSOR EXAMPLE ==========

Original 3d tensor in TensorFlow:
tf.Tensor(
[[[0.40088013 0.02324858]
  [0.87607911 0.91645907]]

 [[0.10591184 0.92419294]
  [0.5397723  0.24957817]]], shape=(2, 2, 2), dtype=float64)

How DAPHNE sees the 3d tensor from TensorFlow:
DenseMatrix(2x4, double)
0.40088 0.0232486 0.876079 0.916459
0.105912 0.924193 0.539772 0.249578

Result of adding 100, back in Python:
tf.Tensor(
[[100.40088013 100.02324858 100.87607911 100.91645907]
 [100.10591184 100.92419294 100.5397723  100.24957817]], shape=(2, 4), dtype=float64)

Result of adding 100, back in Python (with original shape):
tf.Tensor(
[[[100.40088013 100.02324858]
  [100.87607911 100.91645907]]

 [[100.10591184 100.92419294]
  [100.5397723  100.24957817]]], shape=(2, 2, 2), dtype=float64)
```

### Data Exchange with PyTorch

*Example:*

```python
from daphne.context.daphne_context import DaphneContext
import torch
import numpy as np

dc = DaphneContext()

print("========== 2D TENSOR EXAMPLE ==========\n")

# Create data in PyTorch/numpy.
t2d = torch.tensor(np.random.random(size=(2, 4)))

print("Original 2d tensor in PyTorch:")
print(t2d)

# Transfer data to DaphneLib (lazily evaluated).
T2D = dc.from_pytorch(t2d)

print("\nHow DAPHNE sees the 2d tensor from PyTorch:")
T2D.print().compute()

# Add 100 to each value in T2D.
T2D = T2D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T2D.compute(asPyTorch=True))

print("\n========== 3D TENSOR EXAMPLE ==========\n")

# Create data in PyTorch/numpy.
t3d = torch.tensor(np.random.random(size=(2, 2, 2)))

print("Original 3d tensor in PyTorch:")
print(t3d)

# Transfer data to DaphneLib (lazily evaluated).
T3D, T3D_shape = dc.from_pytorch(t3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from PyTorch:")
T3D.print().compute()

# Add 100 to each value in T3D.
T3D = T3D + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100, back in Python:")
print(T3D.compute(asPyTorch=True))
print("\nResult of adding 100, back in Python (with original shape):")
print(T3D.compute(asPyTorch=True, shape=T3D_shape))
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-pytorch.py
```

*Output (random numbers may vary):*

```text
========== 2D TENSOR EXAMPLE ==========

Original 2d tensor in PyTorch:
tensor([[0.1205, 0.8747, 0.1717, 0.0216],
        [0.7999, 0.6932, 0.4386, 0.0873]], dtype=torch.float64)

How DAPHNE sees the 2d tensor from PyTorch:
DenseMatrix(2x4, double)
0.120505 0.874691 0.171693 0.0215546
0.799858 0.693205 0.438637 0.0872659

Result of adding 100, back in Python:
tensor([[100.1205, 100.8747, 100.1717, 100.0216],
        [100.7999, 100.6932, 100.4386, 100.0873]], dtype=torch.float64)

========== 3D TENSOR EXAMPLE ==========

Original 3d tensor in PyTorch:
tensor([[[0.5474, 0.9653],
         [0.7891, 0.0573]],

        [[0.4116, 0.6326],
         [0.3148, 0.3607]]], dtype=torch.float64)

How DAPHNE sees the 3d tensor from PyTorch:
DenseMatrix(2x4, double)
0.547449 0.965315 0.78909 0.0572619
0.411593 0.632629 0.314841 0.360657

Result of adding 100, back in Python:
tensor([[100.5474, 100.9653, 100.7891, 100.0573],
        [100.4116, 100.6326, 100.3148, 100.3607]], dtype=torch.float64)

Result of adding 100, back in Python (with original shape):
tensor([[[100.5474, 100.9653],
         [100.7891, 100.0573]],

        [[100.4116, 100.6326],
         [100.3148, 100.3607]]], dtype=torch.float64)
```

## Known Limitations

DaphneLib is still in an early development stage.
Thus, there are a few limitations that users should be aware of.
We plan to fix all of these limitations in the future.

- Using DAPHNE's command-line arguments to influence its behavior is not supported yet.
- Some DaphneDSL built-in functions are not represented by DaphneLib methods yet.
- High-level primitives for integrated data analysis pipelines, which are implemented in DaphneDSL, cannot be called from DaphneLib yet.
