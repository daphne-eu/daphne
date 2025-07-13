// RUN: /daphne/bin/daphne --explain property_inference /daphne/scripts/examples/mm_RowColTranspose.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

// COM: This test verifies the simplification of (X @ Y)[row, col] to X[row,] @ Y[,col]
// COM: even when X or Y are transposed. 

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 4 : si64} : () -> si64
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = 8 : si64} : () -> si64
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = 8 : index} : () -> index
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = 3 : si64} : () -> si64
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = 7 : si64} : () -> si64
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 1.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C8:.*]] = "daphne.constant"() {value = 2.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C9:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
// CHECK-DAG: %[[C10:.*]] = "daphne.constant"() {value = 1 : si64} : () -> si64

// CHECK: %[[M1:.*]] = "daphne.randMatrix"(%[[C3]], %[[C3]], %[[C7]], %[[C8]], %[[C9]], %[[C10]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<8x8xf64:sp[1.000000e+00]>
// CHECK: %[[M2:.*]] = "daphne.randMatrix"(%[[C3]], %[[C3]], %[[C7]], %[[C8]], %[[C9]], %[[C10]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<8x8xf64:sp[1.000000e+00]>

// CHECK: %[[MM1_SLICE_ROW:.*]] = "daphne.sliceRow"(%[[M1]], %[[C6]], %[[C2]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<1x8xf64>
// CHECK: %[[MM1_SLICE_COL:.*]] = "daphne.sliceCol"(%[[M2]], %[[C5]], %[[C1]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<8x1xf64>
// CHECK: %[[MM1_MATMUL:.*]] = "daphne.matMul"(%[[MM1_SLICE_ROW]], %[[MM1_SLICE_COL]], %[[C0]], %[[C0]]) : (!daphne.Matrix<1x8xf64>, !daphne.Matrix<8x1xf64>, i1, i1) -> !daphne.Matrix<1x1xf64>

// CHECK: %[[M1_TRANSPOSED:.*]] = "daphne.transpose"(%[[M1]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>) -> !daphne.Matrix<8x8xf64:sp[1.000000e+00]>
// CHECK: %[[M2_TRANSPOSED:.*]] = "daphne.transpose"(%[[M2]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>) -> !daphne.Matrix<8x8xf64:sp[1.000000e+00]>
// CHECK: %[[MM2_SLICE_ROW:.*]] = "daphne.sliceRow"(%[[M1_TRANSPOSED]], %[[C6]], %[[C2]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<1x8xf64>
// CHECK: %[[MM2_SLICE_COL:.*]] = "daphne.sliceCol"(%[[M2_TRANSPOSED]], %[[C5]], %[[C1]]) : (!daphne.Matrix<8x8xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<8x1xf64>
// CHECK: %[[MM2_MATMUL:.*]] = "daphne.matMul"(%[[MM2_SLICE_ROW]], %[[MM2_SLICE_COL]], %[[C0]], %[[C0]]) : (!daphne.Matrix<1x8xf64>, !daphne.Matrix<8x1xf64>, i1, i1) -> !daphne.Matrix<1x1xf64>

// CHECK: %[[MM3_MATMUL:.*]] = "daphne.matMul"(%[[MM1_SLICE_ROW]], %[[MM2_SLICE_COL]], %[[C0]], %[[C0]]) : (!daphne.Matrix<1x8xf64>, !daphne.Matrix<8x1xf64>, i1, i1) -> !daphne.Matrix<1x1xf64>

// CHECK: %[[MM4_MATMUL:.*]] = "daphne.matMul"(%[[MM2_SLICE_ROW]], %[[MM1_SLICE_COL]], %[[C0]], %[[C0]]) : (!daphne.Matrix<1x8xf64>, !daphne.Matrix<8x1xf64>, i1, i1) -> !daphne.Matrix<1x1xf64>

// CHECK: "daphne.print"(%[[MM1_MATMUL]], %[[C4]], %[[C0]]) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
// CHECK: "daphne.print"(%[[MM2_MATMUL]], %[[C4]], %[[C0]]) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
// CHECK: "daphne.print"(%[[MM3_MATMUL]], %[[C4]], %[[C0]]) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
// CHECK: "daphne.print"(%[[MM4_MATMUL]], %[[C4]], %[[C0]]) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }