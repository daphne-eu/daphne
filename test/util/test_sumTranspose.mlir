// RUN: /daphne/bin/daphne --explain property_inference /daphne/scripts/examples/sum_transpose.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

// COM: This test verifies the simplification of sum(transpose(X)) to sum(X)

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = 3 : index} : () -> index
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 4 : index} : () -> index
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = 4 : si64} : () -> si64
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = 3.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C1]], %[[C0]], %[[C5]], %[[C6]], %[[C7]], %[[C4]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x3xf64:sp[1.000000e+00]>

// CHECK-NOT: "daphne.transpose" // transpose should be gone

// CHECK: %[[SUM_RES:.*]] = "daphne.sumAll"(%[[M1]]) : (!daphne.Matrix<4x3xf64:sp[1.000000e+00]>) -> f64
// CHECK: "daphne.print"(%[[SUM_RES]], %[[C3]], %[[C2]]) : (f64, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }