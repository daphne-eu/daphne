
// RUN: /daphne/bin/daphne --explain property_inference /daphne/scripts/examples/trace.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

// COM: This test verifies the simplification of: trace(X @ Y) = sum(diagVector(X @ Y)) → sum(X @ transpose(Y))

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = 7 : index} : () -> index
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = 1.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 42 : si64} : () -> si64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C0]], %[[C0]], %[[C3]], %[[C4]], %[[C3]], %[[C5]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<7x7xf64:sp[1.000000e+00]>
// CHECK: %[[M2:.*]] = "daphne.randMatrix"(%[[C0]], %[[C0]], %[[C3]], %[[C4]], %[[C3]], %[[C5]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<7x7xf64:sp[1.000000e+00]>

// CHECK-NOT: "daphne.matMul" // This is the key check: matMul should be gone!
// CHECK-NOT: "daphne.diagVector" // diagVector should also be gone if the simplification happened

// CHECK: %[[TRANSPOSE_M2:.*]] = "daphne.transpose"(%[[M2]]) : (!daphne.Matrix<7x7xf64:sp[1.000000e+00]>) -> !daphne.Matrix<7x7xf64:sp[1.000000e+00]>
// CHECK: %[[EWMUL_RES:.*]] = "daphne.ewMul"(%[[M1]], %[[TRANSPOSE_M2]]) : (!daphne.Matrix<7x7xf64:sp[1.000000e+00]>, !daphne.Matrix<7x7xf64:sp[1.000000e+00]>) -> !daphne.Matrix<7x7xf64:sp[1.000000e+00]>
// CHECK: %[[SUM_RES:.*]] = "daphne.sumAll"(%[[EWMUL_RES]]) : (!daphne.Matrix<7x7xf64:sp[1.000000e+00]>) -> f64
// CHECK: "daphne.print"(%[[SUM_RES]], %[[C1]], %[[C2]]) : (f64, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }