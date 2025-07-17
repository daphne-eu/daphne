
// RUN: /daphne/bin/daphne --explain property_inference /daphne/scripts/examples/dyn_insert.daph 2>&1 | /usr/lib/llvm-18/bin/FileCheck %s

// COM: specifically checking that the full-matrix insertion (Q[0:7,0:8]=Z)

// CHECK: IR after inference:
// CHECK: module {
// CHECK:   func.func @main() {
// CHECK-DAG: %[[C0:.*]] = "daphne.constant"() {value = 8 : index} : () -> index
// CHECK-DAG: %[[C1:.*]] = "daphne.constant"() {value = 7 : index} : () -> index
// CHECK-DAG: %[[C2:.*]] = "daphne.constant"() {value = 4 : index} : () -> index
// CHECK-DAG: %[[C3:.*]] = "daphne.constant"() {value = 3 : index} : () -> index
// CHECK-DAG: %[[C4:.*]] = "daphne.constant"() {value = false} : () -> i1
// CHECK-DAG: %[[C5:.*]] = "daphne.constant"() {value = true} : () -> i1
// CHECK-DAG: %[[C6:.*]] = "daphne.constant"() {value = 6 : si64} : () -> si64
// CHECK-DAG: %[[C7:.*]] = "daphne.constant"() {value = 0 : si64} : () -> si64
// CHECK-DAG: %[[C8:.*]] = "daphne.constant"() {value = 2 : si64} : () -> si64
// CHECK-DAG: %[[C9:.*]] = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
// CHECK-DAG: %[[C10:.*]] = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
// CHECK-DAG: %[[C11:.*]] = "daphne.constant"() {value = 1.000000e+01 : f64} : () -> f64
// CHECK-DAG: %[[C12:.*]] = "daphne.constant"() {value = 3 : si64} : () -> si64
// CHECK-DAG: %[[C13:.*]] = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64

// without simplification
// CHECK: %[[X_FILL:.*]] = "daphne.fill"(%[[C13]], %[[C1]], %[[C0]]) : (f64, index, index) -> !daphne.Matrix<7x8xf64:symmetric[false]>
// CHECK: %[[Y_RAND:.*]] = "daphne.randMatrix"(%[[C3]], %[[C2]], %[[C11]], %[[C10]], %[[C9]], %[[C8]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x4xf64:sp[1.000000e+00]>
// CHECK: %[[SLICE_ROW_1:.*]] = "daphne.sliceRow"(%[[X_FILL]], %[[C7]], %[[C12]]) : (!daphne.Matrix<7x8xf64:symmetric[false]>, si64, si64) -> !daphne.Matrix<3x8xf64>
// CHECK: %[[INSERT_COL_1:.*]] = "daphne.insertCol"(%[[SLICE_ROW_1]], %[[Y_RAND]], %[[C8]], %[[C6]]) : (!daphne.Matrix<3x8xf64>, !daphne.Matrix<3x4xf64:sp[1.000000e+00]>, si64, si64) -> !daphne.Matrix<3x8xf64>
// CHECK: %[[INSERT_ROW_1:.*]] = "daphne.insertRow"(%[[X_FILL]], %[[INSERT_COL_1]], %[[C7]], %[[C12]]) : (!daphne.Matrix<7x8xf64:symmetric[false]>, !daphne.Matrix<3x8xf64>, si64, si64) -> !daphne.Matrix<7x8xf64>
// CHECK: "daphne.print"(%[[Y_RAND]], %[[C5]], %[[C4]]) : (!daphne.Matrix<3x4xf64:sp[1.000000e+00]>, i1, i1) -> ()
// CHECK: "daphne.print"(%[[INSERT_ROW_1]], %[[C5]], %[[C4]]) : (!daphne.Matrix<7x8xf64>, i1, i1) -> ()

// with simplification
// CHECK: %[[Q_FILL:.*]] = "daphne.fill"(%[[C13]], %[[C1]], %[[C0]]) : (f64, index, index) -> !daphne.Matrix<7x8xf64:symmetric[false]>
// CHECK: %[[Z_RAND:.*]] = "daphne.randMatrix"(%[[C1]], %[[C0]], %[[C11]], %[[C10]], %[[C9]], %[[C12]]) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<7x8xf64:sp[1.000000e+00]>
// CHECK-NOT: "daphne.sliceRow"(%[[Q_FILL]], %{{.*}}, %{{.*}})
// CHECK-NOT: "daphne.insertCol"(%{{.*}}, %[[Z_RAND]], %{{.*}}, %{{.*}})
// CHECK-NOT: "daphne.insertRow"(%[[Q_FILL]], %{{.*}}, %{{.*}}, %{{.*}})
// CHECK: "daphne.print"(%[[Z_RAND]], %[[C5]], %[[C4]]) : (!daphne.Matrix<7x8xf64:sp[1.000000e+00]>, i1, i1) -> ()
// CHECK: "daphne.print"(%[[Z_RAND]], %[[C5]], %[[C4]]) : (!daphne.Matrix<7x8xf64:sp[1.000000e+00]>, i1, i1) -> ()
// CHECK: "daphne.return"() : () -> ()
// CHECK: }
// CHECK: }