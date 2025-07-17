// RUN: %daphne_bin--explain property_inference /daphne/scripts/examples/sum_ewAdd.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s


// COM: This test verifies the rewrite for the simplification SUM(X + Y) = SUM(X) + SUM(Y)

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = 1 : index} : () -> index
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 4 : index} : () -> index
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = 100 : si64} : () -> si64
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 0 : si64} : () -> si64
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = 4 : si64} : () -> si64
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C8:.*]] = "daphne.constant"() {value = 2.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C9:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C1]], %[[C0]], %[[C7]], %[[C8]], %[[C9]], %[[C6]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x1xf64:sp[1.000000e+00]>
// CHECK: %[[M2:.*]] = "daphne.randMatrix"(%[[C1]], %[[C0]], %[[C5]], %[[C4]], %[[C9]], %[[C6]]) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<4x1xsi64:sp[1.000000e+00]>
// CHECK: %[[SUM_M1:.*]] = "daphne.sumAll"(%[[M1]]) : (!daphne.Matrix<4x1xf64:sp[1.000000e+00]>) -> f64
// CHECK: %[[SUM_M2:.*]] = "daphne.sumAll"(%[[M2]]) : (!daphne.Matrix<4x1xsi64:sp[1.000000e+00]>) -> f64
// CHECK: %[[ADD_RES:.*]] = arith.addf %[[SUM_M1]], %[[SUM_M2]] : f64
// CHECK: "daphne.print"(%[[ADD_RES]], %[[C3]], %[[C2]]) : (f64, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }
