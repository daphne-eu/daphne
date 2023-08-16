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

DaphneLib will support efficient data exchange with other well-known Python libraries, in both directions.
The data transfer from other Python libraries to DaphneLib can be triggered through the `from_...()` methods of the `DaphneContext` (e.g., `from_numpy()`).
A comprehensive list of these methods can be found in the [DaphneLib API reference](/doc/DaphneLib/APIRef.md#daphnecontext).
The data transfer from DaphneLib back to Python happens during the call to `compute()`.
If the result of the computation in DAPHNE is a matrix, `compute()` returns a `numpy.ndarray`; if the result is a frame, it returns a `pandas.DataFrame`; and if the result is a scalar, it returns a plain Python scalar.

So far, DaphneLib can exchange data with numpy (via shared memory) pandas (via shared memory or CSV files), TensorFlow (via shared memory) and PyTorch (via shared memory).
Furthermore, we are working on making the data exchange more efficient in general.

### Data Exchange with numpy via shared memory

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

### Data Exchange with pandas via shared memory

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

### Data Exchange with tensorflow via shared memory

*Example:*

```python
from api.python.context.daphne_context import DaphneContext
import tensorflow as tf
import numpy as np

dc = DaphneContext()

# 2D Tensor Example
# Example usage for a 3x3 tensor
tensor2d = tf.constant(np.random.randn(4,3))

# Print the tensor
print("How the 2d Tensor looks in Python:")
print(tensor2d)

# Transfer data to DaphneLib (lazily evaluated).
T2D = dc.from_tensorflow(tensor2d)

print("\nHow DAPHNE sees the 2d tensor from tensorflow:")
print(T2D.compute(isTensorflow=True))

# 3D Tensor Example
# Example usage for a 3x3x3 tensor
tensor3d = tf.constant(np.random.randn(4,3,4))

# Print the tensor
print("\nHow the 3d Tensor looks in Python:")
print(tensor3d)

# Transfer data to DaphneLib (lazily evaluated).
T3D, T3D_shape = dc.from_tensorflow(tensor3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from tensorflow:")
print(T3D.compute(isTensorflow=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

print("\nHow the 3d tensor looks transformed back to the original shape:")
tensor3d_back = T3D.compute(isTensorflow=True, shape=T3D_shape)
print(tensor3d_back)

# 4D Tensor Example
# Example usage for a 3x3x3x3 tensor
tensor4d = tf.constant(np.random.randn(3,3,3,3))

# Print the tensor
print("\nHow the 4d Tensor looks in Python:")
print(tensor4d)

# Transfer data to DaphneLib (lazily evaluated).
T4D, T4D_shape = dc.from_tensorflow(tensor4d, return_shape=True)

print("\nHow DAPHNE sees the 4d tensor from tensorflow:")
print(T3D.compute(isTensorflow=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

tensor4d_back = T4D.compute(isTensorflow=True, shape=T4D_shape)
print("\nHow the 4d tensor looks transformed back to the original shape:")
print(tensor4d_back)
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-tensorFlow.py
```

*Output:*

```text
How the 2d Tensor looks in Python:
tf.Tensor(
[[ 1.31308388 -0.36838566 -0.03866165]
 [ 0.25121815  1.2599527  -0.32129125]
 [ 0.49458581 -0.39470534  0.76608223]
 [-0.41603116  0.94447836 -0.14717533]], shape=(4, 3), dtype=float64)

How DAPHNE sees the 2d tensor from tensorflow:
tf.Tensor(
[[ 1.31308388 -0.36838566 -0.03866165]
 [ 0.25121815  1.2599527  -0.32129125]
 [ 0.49458581 -0.39470534  0.76608223]
 [-0.41603116  0.94447836 -0.14717533]], shape=(4, 3), dtype=float64)

How the 3d Tensor looks in Python:
tf.Tensor(
[[[ 1.56864929 -1.22525487  1.69396538 -1.28685487]
  [-0.20661403 -1.24320567  0.39141006  0.63943356]
  [-1.14795786 -0.12232087 -1.18676617  1.93846369]]

 [[ 1.22882572  0.63120536  0.47275368  0.22931509]
  [-0.98383632 -0.63013434  0.59430688 -0.40633607]
  [ 0.33884281 -1.03378493 -1.20880607  0.77959906]]

 [[ 1.2301541  -0.32946876 -1.92364301 -1.16800199]
  [-0.01093835  0.24008057  1.10871988 -0.85863546]
  [ 1.40991172  2.72914074 -0.95619877  1.24676744]]

 [[-1.23989242  0.55945371  1.70992959  0.45362342]
  [ 0.74710677 -0.20035434 -0.43069097 -0.40705533]
  [ 1.72164462  0.05490404 -1.22038438  0.4368896 ]]], shape=(4, 3, 4), dtype=float64)

How DAPHNE sees the 3d tensor from tensorflow:
tf.Tensor(
[[ 1.56864929 -1.22525487  1.69396538 -1.28685487 -0.20661403 -1.24320567
   0.39141006  0.63943356 -1.14795786 -0.12232087 -1.18676617  1.93846369]
 [ 1.22882572  0.63120536  0.47275368  0.22931509 -0.98383632 -0.63013434
   0.59430688 -0.40633607  0.33884281 -1.03378493 -1.20880607  0.77959906]
 [ 1.2301541  -0.32946876 -1.92364301 -1.16800199 -0.01093835  0.24008057
   1.10871988 -0.85863546  1.40991172  2.72914074 -0.95619877  1.24676744]
 [-1.23989242  0.55945371  1.70992959  0.45362342  0.74710677 -0.20035434
  -0.43069097 -0.40705533  1.72164462  0.05490404 -1.22038438  0.4368896 ]], shape=(4, 12), dtype=float64)

How the original shape of the tensor looks like:
(4, 3, 4)

How the 3d tensor looks transformed back to the original shape:
tf.Tensor(
[[[ 1.56864929 -1.22525487  1.69396538 -1.28685487]
  [-0.20661403 -1.24320567  0.39141006  0.63943356]
  [-1.14795786 -0.12232087 -1.18676617  1.93846369]]

 [[ 1.22882572  0.63120536  0.47275368  0.22931509]
  [-0.98383632 -0.63013434  0.59430688 -0.40633607]
  [ 0.33884281 -1.03378493 -1.20880607  0.77959906]]

 [[ 1.2301541  -0.32946876 -1.92364301 -1.16800199]
  [-0.01093835  0.24008057  1.10871988 -0.85863546]
  [ 1.40991172  2.72914074 -0.95619877  1.24676744]]

 [[-1.23989242  0.55945371  1.70992959  0.45362342]
  [ 0.74710677 -0.20035434 -0.43069097 -0.40705533]
  [ 1.72164462  0.05490404 -1.22038438  0.4368896 ]]], shape=(4, 3, 4), dtype=float64)

How the 4d Tensor looks in Python:
tf.Tensor(
[[[[-0.52886132 -0.4704277  -0.18111409]
   [ 1.93197892  0.37123903 -0.82513818]
   [ 0.48800381  0.24264207  1.27609428]]

  [[-1.09389509  1.12551675  0.19489267]
   [-0.07091619  0.13682964  2.18854758]
   [-1.12114595 -0.07260645 -0.35965432]]

  [[ 0.08628853  1.62228194  1.94591204]
   [-0.00529898 -2.1353978   2.21532946]
   [ 1.15400797 -0.21243173 -0.74804815]]]


 [[[-0.71450509 -0.94586477  0.30400512]
   [-1.19002157 -1.32622149  0.38562885]
   [ 0.21558625  1.66783251  0.21270647]]

  [[-0.95164753 -1.9955329  -0.51945703]
   [-2.0206637   0.97127622 -0.63599864]
   [ 0.60782688  0.40940756 -0.05654895]]

  [[ 0.9798393   1.01557608 -0.8309942 ]
   [ 1.51148814 -0.69474533  1.16672767]
   [-0.86577279  0.16231409  1.15357895]]]


 [[[-0.08314909  1.28768699  0.54339417]
   [-0.91566317 -0.81508354 -0.16462286]
   [-0.03589313 -0.6386152  -1.27889382]]

  [[-1.49414496  0.03240697 -2.13704588]
   [ 1.85737484 -0.64363124  1.78605333]
   [-1.07356168  1.12843806 -0.35632122]]

  [[ 1.69598098 -0.92153308 -0.5534102 ]
   [ 0.82215939  0.31581578 -0.76008927]
   [ 0.26870027 -2.31707979  1.22018738]]]], shape=(3, 3, 3, 3), dtype=float64)

How DAPHNE sees the 4d tensor from tensorflow:
tf.Tensor(
[[ 1.56864929 -1.22525487  1.69396538 -1.28685487 -0.20661403 -1.24320567
   0.39141006  0.63943356 -1.14795786 -0.12232087 -1.18676617  1.93846369]
 [ 1.22882572  0.63120536  0.47275368  0.22931509 -0.98383632 -0.63013434
   0.59430688 -0.40633607  0.33884281 -1.03378493 -1.20880607  0.77959906]
 [ 1.2301541  -0.32946876 -1.92364301 -1.16800199 -0.01093835  0.24008057
   1.10871988 -0.85863546  1.40991172  2.72914074 -0.95619877  1.24676744]
 [-1.23989242  0.55945371  1.70992959  0.45362342  0.74710677 -0.20035434
  -0.43069097 -0.40705533  1.72164462  0.05490404 -1.22038438  0.4368896 ]], shape=(4, 12), dtype=float64)

How the original shape of the tensor looks like:
(4, 3, 4)

How the 4d tensor looks transformed back to the original shape:
tf.Tensor(
[[[[-0.52886132 -0.4704277  -0.18111409]
   [ 1.93197892  0.37123903 -0.82513818]
   [ 0.48800381  0.24264207  1.27609428]]

  [[-1.09389509  1.12551675  0.19489267]
   [-0.07091619  0.13682964  2.18854758]
   [-1.12114595 -0.07260645 -0.35965432]]

  [[ 0.08628853  1.62228194  1.94591204]
   [-0.00529898 -2.1353978   2.21532946]
   [ 1.15400797 -0.21243173 -0.74804815]]]


 [[[-0.71450509 -0.94586477  0.30400512]
   [-1.19002157 -1.32622149  0.38562885]
   [ 0.21558625  1.66783251  0.21270647]]

  [[-0.95164753 -1.9955329  -0.51945703]
   [-2.0206637   0.97127622 -0.63599864]
   [ 0.60782688  0.40940756 -0.05654895]]

  [[ 0.9798393   1.01557608 -0.8309942 ]
   [ 1.51148814 -0.69474533  1.16672767]
   [-0.86577279  0.16231409  1.15357895]]]


 [[[-0.08314909  1.28768699  0.54339417]
   [-0.91566317 -0.81508354 -0.16462286]
   [-0.03589313 -0.6386152  -1.27889382]]

  [[-1.49414496  0.03240697 -2.13704588]
   [ 1.85737484 -0.64363124  1.78605333]
   [-1.07356168  1.12843806 -0.35632122]]

  [[ 1.69598098 -0.92153308 -0.5534102 ]
   [ 0.82215939  0.31581578 -0.76008927]
   [ 0.26870027 -2.31707979  1.22018738]]]], shape=(3, 3, 3, 3), dtype=float64)
```

### Data Exchange with pytorch via shared memory

*Example:*

```python
from api.python.context.daphne_context import DaphneContext
import torch
import numpy as np

dc = DaphneContext()

# 2D Tensor Example
# Example usage for a 3x3 tensor
tensor2d = torch.tensor(np.random.randn(4, 3))

# Print the tensor
print("How the 2d Tensor looks in Python:")
print(tensor2d)

# Transfer data to DaphneLib (lazily evaluated).
T2D = dc.from_pytorch(tensor2d)

print("\nHow DAPHNE sees the 2d tensor from PyTorch:")
print(T2D.compute(isPytorch=True))

# 3D Tensor Example
# Example usage for a 3x3x3 tensor
tensor3d = torch.tensor(np.random.randn(4, 3, 4))

# Print the tensor
print("\nHow the 3d Tensor looks in Python:")
print(tensor3d)

# Transfer data to DaphneLib (lazily evaluated).
T3D, T3D_shape = dc.from_pytorch(tensor3d, return_shape=True)

print("\nHow DAPHNE sees the 3d tensor from PyTorch:")
print(T3D.compute(isPytorch=True))

print("\nHow the original shape of the tensor looks like:")
print(T3D_shape)

print("\nHow the 3d tensor looks transformed back to the original shape:")
tensor3d_back = T3D.compute(isPytorch=True, shape=T3D_shape)
print(tensor3d_back)

# 4D Tensor Example
# Example usage for a 3x3x3x3 tensor
tensor4d = torch.tensor(np.random.randn(3, 3, 3, 3))

# Print the tensor
print("\nHow the 4d Tensor looks in Python:")
print(tensor4d)

# Transfer data to DaphneLib (lazily evaluated).
T4D, T4D_shape = dc.from_pytorch(tensor4d, return_shape=True)

print("\nHow DAPHNE sees the 4d tensor from PyTorch:")
print(T4D.compute(isPytorch=True))

print("\nHow the original shape of the tensor looks like:")
print(T4D_shape)

tensor4d_back = T4D.compute(isPytorch=True, shape=T4D_shape)
print("\nHow the 4d tensor looks transformed back to the original shape:")
print(tensor4d_back)
```

*Run by:*

```shell
python3 scripts/examples/daphnelib/data-exchange-pytorch.py
```

*Output:*

```text
How the 2d Tensor looks in Python:
tensor([[-1.8535, -0.7521,  0.9446],
        [-1.1868, -0.2862,  0.0437],
        [-1.3503, -1.0055, -0.3218],
        [ 0.8003,  1.1750,  1.8912]], dtype=torch.float64)

How DAPHNE sees the 2d tensor from PyTorch:
tensor([[-1.8535, -0.7521,  0.9446],
        [-1.1868, -0.2862,  0.0437],
        [-1.3503, -1.0055, -0.3218],
        [ 0.8003,  1.1750,  1.8912]], dtype=torch.float64)

How the 3d Tensor looks in Python:
tensor([[[ 0.1129, -1.2178,  1.3257, -0.6831],
         [ 0.0263, -0.7379, -2.3299, -1.7824],
         [-0.3876, -0.7956, -0.4230,  0.8579]],

        [[-0.3974, -1.0008,  0.0080, -1.2983],
         [ 1.0793,  0.5160,  2.1831, -1.8404],
         [ 0.3937,  0.9983, -0.6532,  0.0913]],

        [[ 1.3041, -0.7766, -1.2593, -0.6739],
         [ 0.5801,  1.2320, -1.0374,  0.8682],
         [-0.3794, -0.5212,  0.0240, -0.6127]],

        [[ 0.3299, -1.8847,  1.4791, -0.4569],
         [-0.6936, -0.2048, -2.0930, -0.8661],
         [ 0.4844,  0.1234,  1.1370, -0.4604]]], dtype=torch.float64)

How DAPHNE sees the 3d tensor from PyTorch:
tensor([[ 0.1129, -1.2178,  1.3257, -0.6831,  0.0263, -0.7379, -2.3299, -1.7824,
         -0.3876, -0.7956, -0.4230,  0.8579],
        [-0.3974, -1.0008,  0.0080, -1.2983,  1.0793,  0.5160,  2.1831, -1.8404,
          0.3937,  0.9983, -0.6532,  0.0913],
        [ 1.3041, -0.7766, -1.2593, -0.6739,  0.5801,  1.2320, -1.0374,  0.8682,
         -0.3794, -0.5212,  0.0240, -0.6127],
        [ 0.3299, -1.8847,  1.4791, -0.4569, -0.6936, -0.2048, -2.0930, -0.8661,
          0.4844,  0.1234,  1.1370, -0.4604]], dtype=torch.float64)

How the original shape of the tensor looks like:
torch.Size([4, 3, 4])

How the 3d tensor looks transformed back to the original shape:
tensor([[[ 0.1129, -1.2178,  1.3257, -0.6831],
         [ 0.0263, -0.7379, -2.3299, -1.7824],
         [-0.3876, -0.7956, -0.4230,  0.8579]],

        [[-0.3974, -1.0008,  0.0080, -1.2983],
         [ 1.0793,  0.5160,  2.1831, -1.8404],
         [ 0.3937,  0.9983, -0.6532,  0.0913]],

        [[ 1.3041, -0.7766, -1.2593, -0.6739],
         [ 0.5801,  1.2320, -1.0374,  0.8682],
         [-0.3794, -0.5212,  0.0240, -0.6127]],

        [[ 0.3299, -1.8847,  1.4791, -0.4569],
         [-0.6936, -0.2048, -2.0930, -0.8661],
         [ 0.4844,  0.1234,  1.1370, -0.4604]]], dtype=torch.float64)

How the 4d Tensor looks in Python:
tensor([[[[-0.5003,  0.4705, -1.0960],
          [-0.1863, -1.1576,  1.1620],
          [-0.3598,  1.9050,  0.7339]],

         [[ 0.0467, -0.1497,  0.0979],
          [-0.1125, -0.0446,  0.1705],
          [ 0.2118, -0.9024,  0.1665]],

         [[ 0.6349,  1.2377, -1.0773],
          [ 0.4502, -2.3486, -0.3322],
          [-1.4077, -1.6028, -0.2382]]],


        [[[ 0.8510,  0.5391,  1.1461],
          [-0.1255, -2.4009, -1.0430],
          [-0.3053,  0.1883, -0.4420]],

         [[-0.4122, -2.0053, -1.9770],
          [ 0.4979,  1.6253, -0.2520],
          [-0.0394,  0.0823, -0.0203]],

         [[-1.7449,  0.0497,  0.5252],
          [ 0.1901, -1.1366, -0.7679],
          [-1.2489,  0.2665, -0.3104]]],


        [[[-0.3290,  1.6993, -0.6693],
          [-0.2678, -0.8967,  1.9205],
          [-0.0950, -0.0924,  0.5839]],

         [[ 0.9265,  0.5786, -1.9191],
          [-1.6201, -0.1819,  0.5333],
          [-1.0152, -0.1366,  0.7897]],

         [[-0.0612,  0.1154, -0.4391],
          [-0.6169, -1.1214,  2.1395],
          [ 0.0515, -0.7475, -1.9374]]]], dtype=torch.float64)

How DAPHNE sees the 4d tensor from PyTorch:
tensor([[-0.5003,  0.4705, -1.0960, -0.1863, -1.1576,  1.1620, -0.3598,  1.9050,
          0.7339,  0.0467, -0.1497,  0.0979, -0.1125, -0.0446,  0.1705,  0.2118,
         -0.9024,  0.1665,  0.6349,  1.2377, -1.0773,  0.4502, -2.3486, -0.3322,
         -1.4077, -1.6028, -0.2382],
        [ 0.8510,  0.5391,  1.1461, -0.1255, -2.4009, -1.0430, -0.3053,  0.1883,
         -0.4420, -0.4122, -2.0053, -1.9770,  0.4979,  1.6253, -0.2520, -0.0394,
          0.0823, -0.0203, -1.7449,  0.0497,  0.5252,  0.1901, -1.1366, -0.7679,
         -1.2489,  0.2665, -0.3104],
        [-0.3290,  1.6993, -0.6693, -0.2678, -0.8967,  1.9205, -0.0950, -0.0924,
          0.5839,  0.9265,  0.5786, -1.9191, -1.6201, -0.1819,  0.5333, -1.0152,
         -0.1366,  0.7897, -0.0612,  0.1154, -0.4391, -0.6169, -1.1214,  2.1395,
          0.0515, -0.7475, -1.9374]], dtype=torch.float64)

How the original shape of the tensor looks like:
torch.Size([3, 3, 3, 3])

How the 4d tensor looks transformed back to the original shape:
tensor([[[[-0.5003,  0.4705, -1.0960],
          [-0.1863, -1.1576,  1.1620],
          [-0.3598,  1.9050,  0.7339]],

         [[ 0.0467, -0.1497,  0.0979],
          [-0.1125, -0.0446,  0.1705],
          [ 0.2118, -0.9024,  0.1665]],

         [[ 0.6349,  1.2377, -1.0773],
          [ 0.4502, -2.3486, -0.3322],
          [-1.4077, -1.6028, -0.2382]]],


        [[[ 0.8510,  0.5391,  1.1461],
          [-0.1255, -2.4009, -1.0430],
          [-0.3053,  0.1883, -0.4420]],

         [[-0.4122, -2.0053, -1.9770],
          [ 0.4979,  1.6253, -0.2520],
          [-0.0394,  0.0823, -0.0203]],

         [[-1.7449,  0.0497,  0.5252],
          [ 0.1901, -1.1366, -0.7679],
          [-1.2489,  0.2665, -0.3104]]],


        [[[-0.3290,  1.6993, -0.6693],
          [-0.2678, -0.8967,  1.9205],
          [-0.0950, -0.0924,  0.5839]],

         [[ 0.9265,  0.5786, -1.9191],
          [-1.6201, -0.1819,  0.5333],
          [-1.0152, -0.1366,  0.7897]],

         [[-0.0612,  0.1154, -0.4391],
          [-0.6169, -1.1214,  2.1395],
          [ 0.0515, -0.7475, -1.9374]]]], dtype=torch.float64)
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
