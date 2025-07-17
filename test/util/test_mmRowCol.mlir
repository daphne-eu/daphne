// RUN: %daphne_bin --explain property_inference /daphne/scripts/examples/mm_RowCol.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

// COM: This test verifies the simplification of (X @ Y)[row, col] to X[row,] @ Y[,col]

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 4 : si64} : () -> si64
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = 8 : si64} : () -> si64
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = 5 : index} : () -> index
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = 8 : index} : () -> index
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 10 : index} : () -> index
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 3 : si64} : () -> si64
// CHECK-DAG: %[[C8:.*]] = "daphne.constant"() {value = 7 : si64} : () -> si64
// CHECK-DAG: %[[C9:.*]] = "daphne.constant"() {value = 1.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C10:.*]] = "daphne.constant"() {value = 2.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C11:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
// CHECK-DAG: %[[C12:.*]] = "daphne.constant"() {value = 1 : si64} : () -> si64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C5]], %[[C4]], %[[C9]], %[[C10]], %[[C11]], %[[C12]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<10x8xf64:sp[1.000000e+00]>
// CHECK: %[[M2:.*]] = "daphne.randMatrix"(%[[C4]], %[[C3]], %[[C9]], %[[C10]], %[[C11]], %[[C12]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<8x5xf64:sp[1.000000e+00]>

// CHECK: %[[SLICE_ROW:.*]] = "daphne.sliceRow"(%[[M1]], %[[C8]], %[[C2]]) : (!daphne.Matrix<10x8xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<1x8xf64>
// CHECK: %[[SLICE_COL:.*]] = "daphne.sliceCol"(%[[M2]], %[[C7]], %[[C1]]) : (!daphne.Matrix<8x5xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<8x1xf64>
// CHECK: %[[MATMUL_RES:.*]] = "daphne.matMul"(%[[SLICE_ROW]], %[[SLICE_COL]], %[[C0]], %[[C0]]) : (!daphne.Matrix<1x8xf64>, !daphne.Matrix<8x1xf64>, i1, i1) -> !daphne.Matrix<1x1xf64>
// CHECK: "daphne.print"(%[[MATMUL_RES]], %[[C6]], %[[C0]]) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }