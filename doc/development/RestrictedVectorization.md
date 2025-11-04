<!--
Copyright 2025 The DAPHNE Consortium

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

# Restricted Vectorization

When `daphne` is invoked with `--vec`, by default, all vectorizable operations (those that have the custom MLIR interface `Vectorizable`) that have at least one matrix argument or result are considered by the `VectorizeComputationsPass` for vectorization, i.e., for being executed in the context of a vectorized pipeline (`VectorizedPipelineOp`).

However, when conducting research in the context of DAPHNE, it can be desirable to apply DAPHNE's vectorized processing only to certain, well-chosen operations.
To this end, there is an experimental feature for restricting the operations considered by the `VectorizeComputationsPass` to those that have a certain boolean MLIR attribute set to `true` (in addition to the conditions that are required by default).
The name of this attribute is stored in `CompilerUtils::ATTR_VEC` (`src/compiler/utils/CompilerUtils.h`); at the time of this writing, it is called `vec`.
By default, this feature is turned off.
It can be turned on by invoking `daphne` with `--vec-restricted`; note that this command-line argument only takes effect if `--vec` (for turning on DAPHNE's vectorized engine) is specified as well.

## Example

Let's say we want to restrict vectorization to `EwSqrtOp` (elementwise square root, `sqrt()` in DaphneDSL).
To this end, we could modify the DaphneDSL parser to attach the necessary MLIR attribute to those operations as follows.
Note that this example mainly serves the purpose of illustrating how to attach the attribute, which can also be done in a compiler pass or elsewhere, and how to invoke DAPHNE and see the effect.

```diff
--- a/src/parser/daphnedsl/DaphneDSLBuiltins.cpp
+++ b/src/parser/daphnedsl/DaphneDSLBuiltins.cpp
@@ -101,7 +101,10 @@ template <class UnaryOp>
 mlir::Value DaphneDSLBuiltins::createUnaryOp(mlir::Location loc, const std::string &func,
                                              const std::vector<mlir::Value> &args) {
     checkNumArgsExact(loc, func, args.size(), 1);
-    return CompilerUtils::retValWithInferredType(builder.create<UnaryOp>(loc, utils.unknownType, args[0]));
+    auto op = builder.create<UnaryOp>(loc, utils.unknownType, args[0]);
+    if (func == "sqrt")
+        op->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
+    return CompilerUtils::retValWithInferredType(op);
 }
```

After re-building DAPHNE, we can observe that only `EwSqrtOp` (but not other vectorizable ops like `EwAbsOp`) get vectorized if `--vec-restricted` is specified.

*File `example.daphne`:*

```r
X = fill(1.23, 3, 3);
print(abs(sqrt(X)));
```

- *Calling `daphne` with `--vec`:*

    ```bash
    bin/daphne --vec --explain vectorized example.daphne
    ```

    The output reveals that both `daphne.ewSqrt` and `daphne.ewAbs` are inside the `daphne.vectorizedPipeline`:

    ```text
    IR after vectorization:
    module {
      func.func @main() {
        %0 = "daphne.constant"() {value = 3 : index} : () -> index
        %1 = "daphne.constant"() {value = -1 : si64} : () -> si64
        %2 = "daphne.constant"() {value = 1 : si64} : () -> si64
        %3 = "daphne.constant"() {value = -1.000000e+00 : f64} : () -> f64
        %4 = "daphne.constant"() {value = false} : () -> i1
        %5 = "daphne.constant"() {value = true} : () -> i1
        %6 = "daphne.constant"() {value = 1.230000e+00 : f64} : () -> f64
        %7 = "daphne.fill"(%6, %0, %0) : (f64, index, index) -> !daphne.Matrix<3x3xf64:symmetric[true]>
        "daphne.transferProperties"(%7, %3, %2) : (!daphne.Matrix<3x3xf64:symmetric[true]>, f64, si64) -> ()
        %8 = "daphne.vectorizedPipeline"(%7, %0, %0) ({
        ^bb0(%arg0: !daphne.Matrix<?x3xf64:symmetric[true]>):
          %9 = "daphne.ewSqrt"(%arg0) {vec = true} : (!daphne.Matrix<?x3xf64:symmetric[true]>) -> !daphne.Matrix<?x?xf64>
          %10 = "daphne.ewAbs"(%9) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
          "daphne.return"(%10) : (!daphne.Matrix<?x?xf64>) -> ()
        }, {
        }) {combines = [1], operand_segment_sizes = array<i32: 1, 1, 1, 0>, splits = [1]} : (!daphne.Matrix<3x3xf64:symmetric[true]>, index, index) -> !daphne.Matrix<3x3xf64>
        "daphne.transferProperties"(%8, %3, %1) : (!daphne.Matrix<3x3xf64>, f64, si64) -> ()
        "daphne.print"(%8, %5, %4) : (!daphne.Matrix<3x3xf64>, i1, i1) -> ()
        "daphne.return"() : () -> ()
      }
    }
    DenseMatrix(3x3, double)
    1.10905 1.10905 1.10905
    1.10905 1.10905 1.10905
    1.10905 1.10905 1.10905
    ```

- *Calling `daphne` with `--vec --vec-restricted`:*

    ```bash
    bin/daphne --vec --vec-restricted --explain vectorized example.daphne
    ```

    The output reveals that only `daphne.ewSqrt` (but not `daphne.ewAbs`) is inside the `daphne.vectorizedPipeline`:

    ```text
    IR after vectorization:
    module {
      func.func @main() {
        %0 = "daphne.constant"() {value = 3 : index} : () -> index
        %1 = "daphne.constant"() {value = -1 : si64} : () -> si64
        %2 = "daphne.constant"() {value = 1 : si64} : () -> si64
        %3 = "daphne.constant"() {value = -1.000000e+00 : f64} : () -> f64
        %4 = "daphne.constant"() {value = false} : () -> i1
        %5 = "daphne.constant"() {value = true} : () -> i1
        %6 = "daphne.constant"() {value = 1.230000e+00 : f64} : () -> f64
        %7 = "daphne.fill"(%6, %0, %0) : (f64, index, index) -> !daphne.Matrix<3x3xf64:symmetric[true]>
        "daphne.transferProperties"(%7, %3, %2) : (!daphne.Matrix<3x3xf64:symmetric[true]>, f64, si64) -> ()
        %8 = "daphne.vectorizedPipeline"(%7, %0, %0) ({
        ^bb0(%arg0: !daphne.Matrix<?x3xf64:symmetric[true]>):
          %10 = "daphne.ewSqrt"(%arg0) {vec = true} : (!daphne.Matrix<?x3xf64:symmetric[true]>) -> !daphne.Matrix<?x?xf64>
          "daphne.return"(%10) : (!daphne.Matrix<?x?xf64>) -> ()
        }, {
        }) {combines = [1], operand_segment_sizes = array<i32: 1, 1, 1, 0>, splits = [1]} : (!daphne.Matrix<3x3xf64:symmetric[true]>, index, index) -> !daphne.Matrix<3x3xf64>
        "daphne.transferProperties"(%8, %3, %1) : (!daphne.Matrix<3x3xf64>, f64, si64) -> ()
        %9 = "daphne.ewAbs"(%8) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x3xf64>
        "daphne.transferProperties"(%9, %3, %1) : (!daphne.Matrix<3x3xf64>, f64, si64) -> ()
        "daphne.print"(%9, %5, %4) : (!daphne.Matrix<3x3xf64>, i1, i1) -> ()
        "daphne.return"() : () -> ()
      }
    }
    DenseMatrix(3x3, double)
    1.10905 1.10905 1.10905
    1.10905 1.10905 1.10905
    1.10905 1.10905 1.10905
    ```