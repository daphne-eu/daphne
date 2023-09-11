# Enabling updating in-place of data objects in kernels

This document outlines the steps necessary for Daphne/Kernel developers to enable updating objects in-place in operation.

In-place updating refers to reusing and overwriting an input data object for the output. This eliminates the need to allocate a new data object for each operation, potentially reducing peak memory consumption and execution times.

**For example:**

```
X = readMatrix("X.csv");
Y = readMatrix("Y.csv");

Z = sqrt(X + Y);
```

Here, for adding two matrices elementwise, it results in the allocation of a new matrix with size of X and the same for calculating element-wise the square root of the intermediate result of X + Y. If X or Y is not used later in the Daphne application's execution, we could think about using either of them for storing the result. Thus avoiding the need for additional memory allocation.

**Using update in-place in DAPHNE:**

If you want to use the in-place feature, you just need to add --update-in-place to the arguments for calling the DAPHNE. This leads to the execution of an additional compiler pass that enables the update in-place. Appropriate JSON configuration can be used:

```json
...
"enable_update_in_place": false,  /* like --update-in-place */
"explain_update_in_place": false, /* like --explain=update_in_place */
...
```

## Background

There are some condition that needs to hold true so that an update in-place is possible:

### Result and the operand are data objects (Frame or Matrix)

If the result or the operands is a e.g. a scalar value, we do not consider them for in-place updates.
(We will only consider DenseMatrices for further analysis. Although CSRMatrices are possible, there does not exist an example or test cases for them.)

### No interdependence of object elements while calculating a result

This is highly dependent on the operation/kernel. For instance, if the operation is performed element-wise (e.g. adding the first element of one matrix to the first element of another matrix), we can store the result directly in the position of the first element of the original matrix. Storing the result directly in the position of the first element of the original matrix is not feasible when the output calculation depends on other elements. For operations like matrix multiplication or convolution (with overlapping kernels), multiple elements need to be accessed and used to compute an element output. Overwriting the old values with the newly calculated one could lead to an incorrect result.

Therefore, only kernels that support this semantic can be updated in-place. Special algorithms may need to be implemented to make this possible (e.g. In-Place Transposition with cycles).

### No Future Use of data object/buffer in the application

Values can only be overridden if they are not used later in the application's execution. Typically, this is possible when an operand is not used as an operand in a succeding operation. However, due to the use of data buffers, views, and frames, the underlying values may be used multiple times in different data objects. Detecting this at compile-time is not trivial, so we rely on both compile-time and runtime analysis.

**Compile-time analysis**

While compiling the DAPHNE application to LLVM instruction, in the Pass [FlagUpdateInPlacePass.cpp](/src/compiler/lowering/FlagUpdateInPlacePass.cpp) we check whether an operand can be used for in-place updates. The operand is typically the result of an another operation. The operand is usually the result of another operation. If this result is not used as an operand in subsequent operations, we can potentially overwrite its values to avoid memory allocation (refer to the example at the beginning).

Furthermore, we need to propagate this information to the kernel call. We accomplish this by adding an additional attribute (e.g., `inPlaceFutureUse = [false, true]`) to the MLIR operation object. Subsequently, in the [RewriteToCallKernelOpPass.cpp](/src/compiler/lowering/RewriteToCallKernelOpPass.cpp), we append a true/false value for each operand at the end of the operand list of an operation. This allows the information to be available at runtime. We still expect to initialize the result operand with a nullptr.

**Runtime analysis**

Inside the kernels, we need to perform specific checks to identify the possibility of in-place operands. This is crucial to verify that the data object and its underlying data buffer are not being used elsewhere. Essentially, we examine their current usage and dependencies, which cannot be identified during compile-time. To perform this check, we examine the *use_count()* of the underlying *shared_ptr* to the values and the number of references to the data object (here DenseMatrix). This ensures that the values are not used by another object or view.
In combination with the compile-time info, we can determine whether the memory of an operand can be overwritten.

## Steps for enabling update in-place for a kernel

To enable the updating in-place for a kernel, we need to employ different steps. We expect that the kernel is already implemented according to [ImplementBuiltinKernel.md](/doc/development/ImplementBuiltinKernel.md).

1. Mark the operation as InPlaceable
2. Change the kernel signature
3. Adapt the knowledge for improving execution

### 1. Mark the operation as InPlaceable

We need to add an MLIR op interface to the operation. This is necessary for the [FlagUpdateInPlacePass.cpp](/src/compiler/lowering/FlagUpdateInPlacePass.cpp) pass to consider using the operands of the specific operation in-place.

This is accompanied by changing the traits of an operation in [DaphneOps.td](/src/ir/daphneir/DaphneOps.td) by adding `DeclareOpInterfaceMethods<InPlaceOpInterface>` to the list. For instance:

```cpp
def Daphne_MyInPlaceOp : Daphne_Op<"myOp", [
    TypeFromFirstArg,
    NumRowsFromArgNumCols,
    DeclareOpInterfaceMethods<InPlaceOpInterface> //<---
]> {
```

Additionally, we need to indicate which operand should be enabled for in-place updating. To achieve this, we must implement a `std::vector<int> getInPlaceOperands()` method that returns a vector of integers representing the positions of the operands to be considered for in-place update. This is done in [DaphneUpdateInPlaceOpInterface.cpp](/src/ir/daphneir/DaphneUpdateInPlaceOpInterface.cpp):

```cpp
std::vector<int> daphne::MyInPlaceOp::getInPlaceOperands() {
        return {0, 2};
}
```

This denotes that for the operation MyInPlaceOp the operand at position 0 and 2 should be put under consideration for in-place update. The `getInPlaceOperands()` method gets used inside the passes.

**Simplification:**

There are C++ macros available for the most common cases (unary and binary operations) with respective positions for operands {0} and {0,1}.

```cpp
// If the operation should only allow the first two operands to be in-placeable.
// {0,1}
IMPL_IN_PLACE_OPERANDS_BINARYOP(MyInPlaceOp)

// If the operation should only allow the first operand to be in-placeable.
// {0}
IMPL_IN_PLACE_OPERANDS_UNARYOP(MyInPlaceOp)
```

### 2. Change the kernel signature

The [kernels.json](/src/runtime/local/kernels/kernels.json) file needs to be updated to new additional boolean values for each operand that could be updated in-place. This is how the information from the compile-time is available at runtime. They need to be placed at end of the list of runtime parameters:

```json
"runtimeParams": [
	{
		"type": "DTRes *&",
		"name": "res"
	},
	{
		"type": "DTLhs *", /* <-- no const */
		"name": "lhs"
        },
	{
		"type": "int",
		"name": "scale"
        },
	{
		"type": "DTRhs *", /* <-- no const */
		"name": "rhs"
        },
	{
		"type": "bool",
		"name": "hasFutureUseLhs"
        },
	{
		"type": "bool",
		"name": "hasFutureUseRhs"
        },
]
```

For consistency, it is advisable to name the additional values `hasFutureUse*(Name_of_object)`.

In case of an already implemented kernel: the specialization templates and function in the kernel header files needs to include the newly defined runtime arguments. For instance:

```cpp
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MyKernel {
    static void apply(DTRes *& res, DTLhs * lhs, int scale, DTRhs * rhs, bool hasFutureUseLhs, bool hasFutureUseRhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void myKernel(DTRes *& res, DTLhs * lhs, int scale, DTRhs * rhs, bool hasFutureUseLhs, bool hasFutureUseRhs, DCTX(ctx)) {
   MyKernel<DTRes, DTLhs, DTRhs>::apply(res, lhs, scale, rhs, hasFutureUseLhs, hasFutureUseRhs, ctx);
}



// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<typename VTres, typename VTlhs, typename VTrhs>
struct MyKernel<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>> {
    static void apply(DenseMatrix<VTres> *& res, DenseMatrix<VTlhs> * lhs, int scale, DenseMatrix<VTrhs> * rhs, bool hasFutureUseLhs, bool hasFutureUseRhs, DCTX(ctx)) {
	...
}
  
```

### 3. Adapt the knowledge for improving execution

The compiler's information passes provide simple boolean values to check if an operand has a future use. To identify the current use and applicability, we can use the general helper functions available in [InPlaceUtils.h](/src/runtime/local/kernels/InPlaceUtils.h):

**isInPlaceable**

Currently, only DenseMatrices are supported. The function returns true if there is no future use and the references to the data object and underlying data buffer are equal to one. This is typically the minimum requirement for every data object operand.

**isValidType**

Checks if the Data Type (int or float) and the dimensions of two matrices is identical. This is necessary in some cases for guranteeing, that the underlying data buffer has enough memory allocated. Allows for direct reuse of the data object (matrix).

**isValidTypeWeak**

Checks if the data type (int or float) is identical and the allocated memory is identical. This is necessary in some cases for guranteeing, that the underlying data buffer has enough memory allocated. Allows for reuse of the underlying data buffer (shared_ptr). Typically a new data object (matrix) needs to be created.

#### Example

API: To enable the use of pre-allocated memory from the function caller, we need to check if the result pointer is a nullptr before making any decisions.

```cpp
if(res == nullptr) {
	// Check if we can utilize the allocated memory of the lhs matrix for the result.
        // We assume that the result has the same type as lhs (no need to check isValidType).
	if(InPlaceUtils::isInPlaceable(lhs, hasFutureUseLhs)) {
        	res = lhs;
                res->increaseRefCounter();
    	}
	// Check if we can utilize the allocated memory of the rhs matrix for the result.
        // Here we need to check if rhs has a valid type, as it could differ from the result type.
        // E.g. lhs is rectangular and rhs is a column/row vector, the result will be rectangular (rhs cannot store the result).
       	else if(InPlaceUtils::isInPlaceable(rhs, hasFutureUseRhs) && InPlaceUtils::isValidTypeWeak(lhs, rhs)) {
		// If the rhs has the same dimensions as the lhs, we can reuse the rhs matrix object.
            	if(InPlaceUtils::isValidType(lhs, rhs)) {
                    res = rhs;
                    res->increaseRefCounter();
                }
		// As it atleast weak, we can reuse the underlying data buffer of rhs.
                else {
                    res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, rhs->getValues());
                }
            }
	// Otherwise we create a new matrix based on the dimensions of the lhs matrix.
      	else {
		res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, false);
     	}
}
```

This is just an example. The decision on how to use one of the operands depends heavily on the nature of the kernel and its functionality. You need to consider the employed algorithm, as it must also support the in-place semantic.

## Special Notes

It is advisable to adapt the tests accordingly as well and implement new test cases for the use of in-place update.

**Currently supported Kernels:**

| Kernel Name    | Dense | CSR | Frame |
| -------------- | ----- | --- | ----- |
| EwBinaryMat    | ✅    | ❌  |       |
| EwBinaryObjMat | ✅    | ❌  | ✅    |
| EwUnaryMat     | ✅    | ❌  |       |
| Transpose      | ✅    | ❌  |       |

None of these kernels currently support in-place updates when using CUDA or FPGA.

Additional kernels/operations that could support in-place updates:

* InsertColOp, InsertRowOp
* SliceColOp, SliceRowOp
* ReverseOp
* VectorizedPipelineOp
