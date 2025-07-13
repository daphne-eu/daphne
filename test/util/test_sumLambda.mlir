// RUN: /daphne/bin/daphne --explain property_inference /daphne/scripts/examples/sum_lambdaMul.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

//COM: Verification test for the simplification rewrite: SUM(lambda * X) = lambda * SUM(X)


// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = 1 : index} : () -> index
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 3 : index} : () -> index
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = 4 : index} : () -> index
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 42 : si64} : () -> si64
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = 4 : si64} : () -> si64
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C8:.*]] = "daphne.constant"() {value = 3.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C9:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C2]], %[[C1]], %[[C7]], %[[C8]], %[[C9]], %[[C6]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x3xf64:sp[1.000000e+00]>
// CHECK: %[[LAMBDA_MAT_ORIG:.*]] = "daphne.randMatrix"(%[[C0]], %[[C0]], %[[C9]], %[[C7]], %[[C9]], %[[C5]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<1x1xf64:sp[1.000000e+00]>
// CHECK: %[[LAMBDA_MAT_AS_SCALAR:.*]] = "daphne.randMatrix"(%[[C0]], %[[C0]], %[[C9]], %[[C7]], %[[C9]], %[[C5]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<1x1xf64:sp[1.000000e+00]>
// CHECK: %[[LAMBDA_SCALAR:.*]] = "daphne.cast"(%[[LAMBDA_MAT_AS_SCALAR]]) : (!daphne.Matrix<1x1xf64:sp[1.000000e+00]>) -> f64
// CHECK: %[[SUM_M1:.*]] = "daphne.sumAll"(%[[M1]]) : (!daphne.Matrix<4x3xf64:sp[1.000000e+00]>) -> f64
// CHECK: %[[MUL_RES:.*]] = arith.mulf %[[LAMBDA_SCALAR]], %[[SUM_M1]] : f64
// CHECK: "daphne.print"(%[[MUL_RES]], %[[C4]], %[[C3]]) : (f64, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }