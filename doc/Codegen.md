# Code Generation with MLIR

This document describes the process of directly generating code with the MLIR
framework.

## Motivation

DAPHNE provides a kernel for (almost) every DaphneIR operation which reside in
`src/runtime/local/kernels/`. These are precompiled as a shared library and
linked during compile-time. Even though these kernels can be highly optimized
and thus achieve great runtime characteristics, they may not provide a desired
level of extensibility for custom value types. They may also be lacking
information only available at compile-time that could enable further
optimizations. Additionally, through the process of progressively lowering the
input IR, the code generation pipeline may enable more optimization
possibilities such as operator or loop fusion.


As an alternative way to implement our operators we provide the code generation
pipeline which progressively lowers the DaphneIR available after parsing the
DaphneDSL script to operations in either the same dialect or operations from
other dialects. With that, we can optionally replace certain kernels by
generating code directly, and also perform a hybrid compilation approach where
we mix kernel calls with code generation in order to exploit advantages of
both, precompiled kernel libraries and code generation. Code generation passes
are found in `src/compiler/lowering/`.


## Guidelines

Currently, the code generation pipeline is enabled with the CLI flag
`--mlir-codegen`. This adds the following passes that perform transformations and
lowerings:

- [DenseMatrixOptPass](src/compiler/lowering/DaphneOptPass.cpp)
- [MatMulOpLoweringPass](src/compiler/lowering/MatMulOpLowering.cpp)
- [AggAllLoweringPass](src/compiler/lowering/AggAllOpLowering.cpp)
- [MapOpLoweringPass](src/compiler/lowering/MapOpLowering.cpp)
- InlinerPass
- [LowerEwOpPass](src/compiler/lowering/EwOpsLowering.cpp)
- ConvertMathToLLVMPass
- [ModOpLoweringPass](src/compiler/lowering/ModOpLowering.cpp)
- Canonicalizer
- CSE
- LoopFusion
- AffineScalarReplacement
- LowerAffinePass

These passes are added in the `DaphneIrExecutor::buildCodegenPipeline`
function. The `--mlir-hybrid-codegen` flag disables the `MatMulOpLoweringPass` since the
kernel implementation vastly outperforms the generated code of this pass.


#### Runtime Interoperability

Runtime interoperability with the `DenseMatrix` object is achieved with two
kernels in `src/runtime/local/kernels/ConvertDenseMatrixToMemRef.h` and
`src/runtime/local/kernels/ConvertMemRefToDenseMatrix.h` and the corresponding
DaphneOps `Daphne_ConvertMemRefToDenseMatrix` and
`Daphne_ConvertDenseMatrixToMemRef`. These kernels define how a MemRef is
passed to a kernel and how a kernel can return a `StridedMemRefType`.


#### Debugging

In order to enable our debug `PrintIRPass` pass, one has to add `--explain
mlir_code_gen` when running `daphne`. Additionally, it is recommended to use the
`daphne-opt` tool to test passes in isolation. One just has to provide the
input IR for a pass to `daphne-opt` and the correct flag to run the pass (or
multiple passes) on the IR. `daphne-opt` provides all the functionality of the
`mlir-opt` tool.

`daphne-opt --lower-ew --debug-only=dialect-conversion ew.mlir` performs the
`LowerEwOpPass` on the input file `ew.mlir` while providing dialect conversion
debug information.



#### Testing

To test the generated code, there currently are two different approaches.

End-to-end tests can be found under `test/api/cli/codegen/` and are part of the
existing Catch2 test-suite with the its own tag, `TAG_CODEGEN`.

Additionally, there are tests that check the generated IR by running the
`llvm-lit`, `daphne-opt`, and `FileCheck` utilities. These tests reside under
`test/compiler/lowering/`. They are `.mlir` files containing the input IR of a
certain pass, or pass pipeline, and the `llvm-lit` directive at the top of the
file (`RUN:`). In that line we specify how `llvm-lit` executes the test, e.g.,
`// RUN: daphne-opt --lower-ew %s | FileCheck %s`, means that `daphne-opt` is
called with the `--lower-ew` flag and the current file as input, the output of
that, in addition to the file itself, is piped to `FileCheck`. `FileCheck` uses
the comments in the `.mlir` file to check for certain conditions, e.g., `//
CHECK-NOT: daphne.ewAdd` looks through the IR and fails if `daphne.ewAdd` can be
found. These `llvm-lit` tests are all run by the `codegen` testcase in
`test/codegen/Codegen.cpp`.


All codegen tests can be executed by running `bin/run_tests '[codegen]'`.
