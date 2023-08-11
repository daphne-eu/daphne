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

# Implementing a Built-in Kernel for a DaphneIR Operation

## Background

(Almost) every DaphneIR operation will be backed by a kernel (= physical operator) at run-time.
Extensibility w.r.t. kernels is one on the core goals of the DAPHNE system.
It shall be easy for a user to add a custom kernel.
However, the system will offer a full set of built-in kernels so that all DaphneIR operations can be used out-of-the-box.

## Scope

This document focuses on:

- default built-in kernels (not custom/external kernels)
- implementations for CPU (not HW accelerators)
- local execution (not distributed)
- pre-compiled kernels (not on-the-fly code generation and operator fusion)

We will extend the system to those more advanced aspects step by step.

### Guidelines

The following are some guidelines and best practices (rather than strict rules) for implementing built-in kernels.
As we proceed, we might adapt these guidelines step by step.
The goal is to clarify how to implement a built-in kernel and what the thoughts behind it are.
This is meant as a proposal, *comments/suggestions are always welcome*.

**Integration into the directory tree:**

The implementations of built-in kernels shall reside in `src/runtime/local/kernels`.
By default, one C++ header-file should be used for all specializations of a kernel.
Depending on the amount of code, the separation into multiple header files is also possible.
At least, we should rather not mix kernels of different DaphneIR operations in one header file.

**Interfaces:**

Technically, a kernel is a C++ function taking one or more data objects (matrices, frames) and/or scalars as input and returning one or more data objects and/or scalars as output.
As a central idea, (almost) all DaphneIR operations should be able to process (almost) all kinds of Daphne data structures, whereby these could have any Daphne value type.
For example, elementwise binary operations (`+`, `*`, ...) should be applicable to `DenseMatrix` of `double`, `uint32_t`, etc. as well as to `CSRMatrix` of `double`, `uint32_t`, etc. and so on.
This type flexibility motivates the use of C++ template metaprogramming.

Thus, a kernel is a template function with:

- *template parameters*
  - one for the type of each input/output data object (and scalar, if necessary)
- *inputs*
  - of type `const DT *` for data objects (whereby `DT` is a particular C++ type such as `DenseMatrix<double>`)
  - of type `VT` for scalars of some value type `VT`
- *outputs*
  - as a return value, in case of a single scalar
  - as a parameter of type `DT *&` in case of data objects (all output parameters *before* all input parameters)

The reason for passing output data objects as parameters is that this mechanism could be used for efficient update-in-place operations.

The declaration of a kernel function could look as follows:

```cpp
template<class DTRes, class DTArg, typename VT>
void someOp(DTRes *& res, const DTArg * arg, VT otherArg);
```

For most DaphneIR operations, it will not be possible to define the template as a fully generic algorithm.
For example, the algorithms for processing different *data type* implementations (like `DenseMatrix` and `CSRMatrix`) will usually differ significantly.
Thus, we will usually want to specialize the template.
At the same time, it will be possible to implement algorithms generically w.r.t. the *value type*.
Thus, we will usually not need to specialize for that.
Hence, we will often use partial template specialization.
Unfortunately, this is not possible for functions in C++.
Therefore, we use a very typical workaround:
In addition to the kernel template function, we declare a template struct/class with the same template parameters as the kernel function.
By convention, let us call this struct like the kernel function, but starting with upper case letter.
This struct/class has a single static member function with the same results and parameters as the kernel function.
By convention, let us always call this function `apply`.
Then, the kernel function always forwards the processing to the `apply`-function of the correct template instantiation of the kernel struct.
Instead of partially specializing the kernel function, we partially specialize the kernel struct and implement the respective algorithm in that specialization's `apply`-function.
Finally, callers will call an instantiation of the kernel template function.
C++ templates offer many ways to express such (partial) specializations, some examples are given below:

```cpp
// Kernel struct to enable partial template specialization.
template<class DTRes, class DTArg, typename VT>
struct SomeOp {
    static void apply(DTRes *& res, const DTArg * arg, VT otherArg) = delete;
};

// Kernel function to be called from outside.
template<class DTRes, class DTArg, typename VT>
void someOp(DTRes *& res, const DTArg * arg, VT otherArg) {
    SomeOp<DTRes, DTArg, VT>::apply(res, arg, otherArg);
}

// (Partial) specializations of the kernel struct (some examples).

// E.g. for DenseMatrix of any value type.
template<typename VT>
struct SomeOp<DenseMatrix<VT>, DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, VT otherArg) {
        // do something
    }
};

// E.g. for DenseMatrix and CSRMatrix of the same value type.
template<typename VT>
struct SomeOp<DenseMatrix<VT>, CSRMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * arg, VT otherArg) {
        // do something
    }
};

// E.g. for DenseMatrix of independent value types.
template<typename VTRes, typename VTArg, typename VTOtherArg>
struct SomeOp<DenseMatrix<VTRes>, DenseMatrix<VTArg>, VTOtherArg> {
    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, VTOtherArg otherArg) {
        // do something
    }
};

// E.g. for super-class Matrix of the same value type.
template<typename VT>
struct SomeOp<Matrix<VT>, Matrix<VT>, VT> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * arg, VT otherArg) {
        // do something
    }
};

// E.g. for DenseMatrix<double>, CSRMatrix<float>, and double (full specialization).
template<>
struct SomeOp<DenseMatrix<double>, CSRMatrix<float>, double> {
    static void apply(DenseMatrix<double> *& res, const CSRMatrix<float> * arg, double otherArg) {
        // do something
    }
};
```

**Implementation of the `apply`-functions:**

As stated above, the `apply`-function contain the actual implementation of the kernel.
Of course, that depends on what the kernel is supposed to do, but there some recurring actions.

- *Obtaining an output data object*

  Data objects like matrices and frames cannot be obtained using the `new`-operator, but must be obtained from the `DataObjectFactory`, e.g., as follows:

  ```cpp
  auto res = DataObjectFactory::create<CSRMatrix<double>>(3, 4, 6, false);
  ```

  Internally, this `create`-function calls a private constructor of the specified data type implementation; so please have a look at these.
- *Accessing the input and output data objects*

  For efficiency reasons, accessing the data in way specific to the data type implementation is preferred to generic access method of the super-classes.
  That is, if possible, rather use `getValues()` (for `DenseMatrix`) or `getValues()`/`colColIdxs()`/`getRowOffsets()` (for `CSRMatrix`) rather than `get()`/`set()`/`append()` from the super-class `Matrix`.
  The reason is that these generic access methods can incur a lot of unnecessary effort, depending on the data type implementation.
  However, in the end it is always a trade-off between performance and code complexity.
  For kernels that are rarely used or typically used on small data objects, a simple but inefficient implementation might be okay.
  Nevertheless, since the DAPHNE system should be able to handle unexpected scripts efficiently, we should not get too much used to sacrificing efficiency.

### Concrete Examples

For concrete examples, please have a look at existing kernel implementations in [src/runtime/local/kernels](/src/runtime/local/kernels).
For instance, the following kernels represent some interesting cases:

- [ewBinarySca](/src/runtime/local/kernels/EwBinarySca.h) works only on scalars.
- [ewBinaryMat](/src/runtime/local/kernels/EwBinaryMat.h) works only on matrices.
- [ewBinaryObjSca](/src/runtime/local/kernels/EwBinaryObjSca.h) combines matrix/frame and scalar inputs.
- [matMul](/src/runtime/local/kernels/MatMul.h) delegates to an external library (OpenBLAS).

### Test Cases

Implementing test cases for each kernel is important to reduce the likelihood of bugs, now and after changes to the code base.
Please have a look at test cases for existing kernel implementations in [test/runtime/local/kernels](/test/runtime/local/kernels) (surely, these could still be improved).
