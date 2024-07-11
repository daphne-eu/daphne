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

# Porting Numpy to DaphneLib in Python Scripts

A large part of the operations offered by Numpy can already be expressed in DaphneLib.
By porting Numpy-based Python scripts to DaphneLib-based Python scripts, existing workflows can be executed in DAPHNE.
In some cases, this translation is straightforward, especially when a Numpy operation can be replaced by a DaphneLib operation with the same name.
However, in some cases this mapping may not be obvious.
Some Numpy operations must be replaced by a single DaphneLib operation with a different name or by a more general/specialized operation.
Some Numpy operations must be replaced by a complex expression of DaphneLib operations.

In the following, we provide a few **non-exhaustive guidelines** on porting Numpy-based Python scripts to DaphneLib-based Python scripts.
These guidelines represent some cases we have stumbled upon so far.
However, we hope that they convey some general insights into how to approach the translation.
In general, the DaphneLib API is subject to future changes, so things that are difficult at the moment may become easier in the future.

## Some Noteworthy Differences of Numpy and DaphneLib

**Without claim of completeness.**

- Numpy supports *n*-dimensional data objects.
    DAPHNE natively only supports *2*-dimensional data objects.
    *1*-d data objects can be represented in DAPHNE as *2*-d objects where one dimension is *1*.
    General *n*-d objects can be represented in DAPHNE as *2*-d objects where all but one dimension are linearized into the second dimension.
    Furthermore, native support for *n*-d tensors is work-in-progress in DAPHNE.

- Both Numpy and DAPHNE support various types for the elements of a matrix.
    These are called *dtypes* in Numpy and *value types* in DAPHNE.
    Numpy's dtypes can be mapped to DAPHNE value types with the following table:

    Numpy dtype | DAPHNE vtype
    ----- | -----
    `np.float64` | `"f64"`
    `np.float32` | `"f32"`
    `np.int64` | `"si64"`
    `np.int32` | `"si32"`
    `np.int8` | `"si8"`
    `np.uint64` | `"ui64"`
    `np.uint32` | `"ui32"`
    `np.uint8` | `"ui8"`
    
- The Numpy API offers access to low-level aspects that DAPHNE does not, because DAPHNE decides (or will decide) these points automatically to optimize the program.
    Examples include: uninitialized data, the memory layout/order (e.g., C, F, ...), device placement, providing an output object, etc.
    DAPHNE supports (or will support) hints for expert users to *optionally* make such decision by hand.
    Such hints are already partly supported in DaphneDSL, but not supported in DaphneLib yet.

## Concrete Example Operations

**In the following, we assume the Numpy 2.0 API.**
Porting for other Numpy versions may be possible by following similar lines of thought.

- `numpy.`**`astype`**`(x, dtype, /, *, copy=True)`

    *Parameters*
    
    - `x`: supported
    - `dtype`: supported
        - Numpy dtypes are mapped to DAPHNE value types using the table above.
    - `copy`: not supported

    *Example*

    - Numpy

        ```python
        Y = X.astype(np.float32)
        ```

    - DaphneLib

        ```python
        Y = X.asType(vtype="f32")
        ```

- `numpy.`**`column_stack`**`(tup)`

    *Parameters*
    
    - `tup`: supported

    *Example*

    - Numpy

        ```python
        Y = np.column_stack((X1, X2, X3))
        ```

    - DaphneLib

        ```python
        # Ensure that X1, X2, X3 are column matrices.
        X1c = X1.reshape(X1.ncell(), 1)
        X2c = X2.reshape(X2.ncell(), 1)
        X3c = X3.reshape(X3.ncell(), 1)

        Y = X1c.cbind(X2c).cbind(X3c)
        ```

- `numpy.`**`concatenate`**`((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")`

    *Parameters*
    
    - `(a1, a2, ...)`: supported
        - only two at a time
    - `axis`: supported
        - via different operation names (`rbind`/`cbind`)
    - `out`: not supported
    - `dtype`: supported
        - *documentation coming later*

    *Example 1*

    - Numpy

        ```python
        Y = np.concatenate((X1, X2, X3), axis=1)
        ```

    - DaphneLib

        ```python
        Y = X1.cbind(X2).cbind(X3)
        ```

    *Example 2*

    - Numpy

        ```python
        Y = np.concatenate((X1, X2), axis=0)
        ```

    - DaphneLib

        ```python
        Y = X1.rbind(X2)
        ```

- `numpy.`**`copy`**`(a, order='K', subok=False)`
    
    Explicit copying is generally not supported/necessary in DAPHNE.
    Editable references to a data object are not possible in DAPHNE, since references are copy-on-write.

- `numpy.`**`empty`**`(shape, dtype=float, order='C', *, device=None, like=None)`

    Explicitly creating an uninitialized matrix is not possible in DAPHNE.
    As a workaround, one could create an initialized matrix.

    *Parameters*
    
    - `shape`: supported
    - `dtype`: supported
        - Numpy dtypes are mapped to DAPHNE value types by the table above.
    - `order`: not supported
    - `device`: not supported
    - `like`: not supported

    *Example*

    - Numpy

        ```python
        Y = np.empty(shape=(3, 2), dtype=np.float32)
        ```

    - DaphneLib

        ```python
        Y = dc.fill(0, 3, 2).asType(vtype="f32")
        ```

- `numpy.`**`full`**`(shape, fill_value, dtype=None, order='C', *, device=None, like=None)`

    *Parameters*
    
    - `shape`: supported
        - only 2d shapes like `(m, n)`
        - currently bug in DaphneLib when at least one dimension is `0`
    - `fill_value`: supported
    - `dtype`: supported
        - *documentation coming later*
    - `order`: not supported
        - deliberately no direct control in DAPHNE
    - `device`: not supported
        - deliberately no direct control in DAPHNE
    - `like`: not supported

    *Example*

    - Numpy

        ```python
        Y = np.full((m, n), v)
        ```

    - DaphneLib

        ```python
        Y = dc.fill(v, m, n)
        ```

- `numpy.`**`isnan`**`(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature]) = <ufunc 'isnan'>`

    *Parameters*
    
    - `x`: supported
    - *all other parameters*: not supported

    *Example*

    - Numpy

        ```python
        Y = np.isnan(X)
        ```

    - DaphneLib

        ```python
        Y = X.isNan()
        ```

- `numpy.`**`logical_not`**`(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature]) = <ufunc 'logical_not'>`

    <!-- TODO Support exchange of boolean data with numpy. -->
    Will be directly supported in DaphneLib in the future.

    *Parameters*
    
    - `x`: supported
    - *all other parameters*: not supported

    *Example*

    - We assume that `X` is of dtype `np.bool`.
        DaphneLib doesn't support the exchange of boolean arrays with Numpy yet.
        Thus, for the DaphneLib variant, `X` must be changed to an integral dtype before the transfer to DaphneLib, e.g., by `X = X.astype(np.int64)`;
        and the result must be changed back to boolean dtype, e.g., by `Y = Y.astype(np.bool)`.

    - Numpy

        ```python
        Y = np.logical_not(X)
        ```

    - DaphneLib

        ```python
        Y = (X - 1).abs()
        ```

- `numpy.`**`random.choice`**`(a, size=None, replace=True, p=None)`

    *Parameters*
    
    - `a`: supported
        - Note that all DAPHNE data objects are 2d.
    - `size`: supported
        - Only 2d shapes are supported.
    - `replace`: supported
    - `p`: not supported
        - Could be achieved with workarounds. **TODO**

    *Example 1* (draw 1d array from arange without replacement)

    - Numpy

        ```python
        Y = np.random.choice(10, 5, False)
        ```

    - DaphneLib

        ```python
        Y = dc.sample(10, 5, False)
        ```

    - Note that Numpy returns a 1d array, while DAPHNE returns a 5x1 column matrix.

    *Example 2* (draw 2d array from 1d array with replacement)

    - Note that this example doesn't work in DaphneLib yet, since right indexing is still missing (see #657).

    - Numpy

        ```python
        Y = np.random.choice(X, (3, 2), True)
        ```

    - DaphneLib

        ```python
        # Assumes that `X` is a column (m x 1) matrix.
        Y = X[dc.sample(X.nrow(), 3 * 2, True), ].reshape(3, 2)
        ```

- `numpy.`**`random.permutation`**`(x)`

    *Note: Doesn't work in DaphneLib yet, since right indexing is still missing (see #657).*

    *Parameters*
    
    - `x`: supported
        - only 2d arrays/matrices are supported

    *Example*

    - Numpy

        ```python
        Y = np.random.permutation(X)
        ```

    - DaphneLib

        ```python
        Y = X[dc.sample(X.nrow(), X.rnow(), False), ]
        ```

- `numpy.`**`repeat`**`(a, repeats, axis=None)`

    *Parameters*
    
    - `a`: supported
    - `repeats`: supported
        - only scalars are supported
    - `axis`: supported
        - only `None`, `0`, and `1` are supported

    *Example 1* (repeat column matrix along row dimension)

    - Assuming `X` is a column *(m x 1)* matrix.

    - Numpy

        ```python
        Y = np.repeat(X, 3, 0)
        ```

    - DaphneLib

        ```python
        Y = X.outerAdd(dc.fill(0, 1, 3)).reshape(X.nrow() * 3, 1)
        ```

    *Example 2* (repeat row matrix along row dimension)

    - Assuming `X` is a row *(1 x n)* matrix.

    - Numpy

        ```python
        Y = np.repeat(X, 3, 0)
        ```

    - DaphneLib

        ```python
        Y = dc.fill(0, 3, 1).outerAdd(X)
        ```

- `numpy.`**`tile`**`(A, reps)`

    *Note that this doesn't work in DaphneLib yet, as right indexing is still missing (see #657).*

    *Parameters*
    
    - `A`: supported
        - Only 2d matrices are supported.
    - `reps`: supported
        - Only scalars and 2x1 matrices are supported, i.e., repetition along row dimension or row and column dimension.

    *Example*

    - Numpy

        ```python
        Y = np.tile(X, (3, 2))
        ```

    - DaphneLib

        ```python
        Y = X[dc.seq(0, X.nrow() * 3).mod(X.nrow()), dc.seq(0, X.ncol() * 2).mod(X.ncol())]
        ```

<!-- TODO

prioritized:
array
std
median
linalg.norm

where (SQL)
abs
min
max
power
zeros
reshape
argmin
any
mean
random.random

--->