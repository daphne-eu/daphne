// RUN: daphne-opt --record-properties %s | FileCheck %s --check-prefix=CHECK-RECORDED

module {
  func.func @"some_calculation-1-1"(%arg0: !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64> {
    %0 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %2 = "daphne.ewMul"(%arg0, %1) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    %3 = "daphne.ewAdd"(%2, %0) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    "daphne.return"(%3) : (!daphne.Matrix<?x?xsi64>) -> ()
  }
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 11 : index} : () -> index
    %2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 5000 : index} : () -> index
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 42 : si64} : () -> si64
    %10 = "daphne.randMatrix"(%3, %3, %6, %7, %8, %9) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
    %11 = "daphne.randMatrix"(%3, %3, %6, %7, %8, %9) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
    %12 = "daphne.ewAdd"(%10, %11) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
    %13 = "daphne.ewMul"(%12, %2) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
    %14 = "daphne.cast"(%12) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64>
    %15 = "daphne.cast"(%13) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64>
    %16 = "daphne.cast"(%14) : (!daphne.Matrix<5000x5000xsi64>) -> !daphne.Matrix<?x?xsi64>
    %17 = "daphne.cast"(%15) : (!daphne.Matrix<5000x5000xsi64>) -> !daphne.Matrix<?x?xsi64>
    %18:2 = scf.for %arg0 = %0 to %1 step %0 iter_args(%arg1 = %16, %arg2 = %17) -> (!daphne.Matrix<?x?xsi64>, !daphne.Matrix<?x?xsi64>) {
      %21 = "daphne.cast"(%arg0) : (index) -> si64
      %22 = "daphne.generic_call"(%arg1) {callee = "some_calculation-1-1"} : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
      %23 = "daphne.generic_call"(%arg2) {callee = "some_calculation-1-1"} : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
      scf.yield %22, %23 : !daphne.Matrix<?x?xsi64>, !daphne.Matrix<?x?xsi64>
    }
    %19 = "daphne.sumAll"(%18#0) : (!daphne.Matrix<?x?xsi64>) -> si64
    "daphne.print"(%19, %5, %4) : (si64, i1, i1) -> ()
    %20 = "daphne.sumAll"(%18#1) : (!daphne.Matrix<?x?xsi64>) -> si64
    "daphne.print"(%20, %5, %4) : (si64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK-RECORDED: func.func @"some_calculation-1-1"(%arg0: !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64> {{.*}}
// CHECK-RECORDED: "daphne.randMatrix"(%3, %3, %6, %7, %8, %9) {daphne.value_ids = [1 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-RECORDED: "daphne.recordProperties"(%10, %11) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, ui32) -> ()
// CHECK-RECORDED: "daphne.randMatrix"(%3, %3, %6, %7, %8, %9) {daphne.value_ids = [2 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-RECORDED: "daphne.recordProperties"(%12, %13) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, ui32) -> ()
// CHECK-RECORDED: "daphne.ewAdd"(%10, %12) {daphne.value_ids = [3 : ui32]} : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-RECORDED: "daphne.recordProperties"(%14, %15) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, ui32) -> ()
// CHECK-RECORDED: "daphne.ewMul"(%14, %2) {daphne.value_ids = [4 : ui32]} : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-RECORDED: "daphne.recordProperties"(%16, %17) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, ui32) -> ()
// CHECK-RECORDED: "daphne.cast"(%14) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64>
// CHECK-RECORDED: "daphne.cast"(%16) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64>
// CHECK-RECORDED: "daphne.cast"(%18) : (!daphne.Matrix<5000x5000xsi64>) -> !daphne.Matrix<?x?xsi64>
// CHECK-RECORDED: "daphne.cast"(%19) : (!daphne.Matrix<5000x5000xsi64>) -> !daphne.Matrix<?x?xsi64>
// CHECK-RECORDED: scf.for {{.*}}
// CHECK-RECORDED: scf.yield %28, %29
// CHECK_RECORDED: {daphne.value_ids = [5 : ui32, 6 : ui32]}
// CHECK-RECORDED: "daphne.recordProperties"(%22#1, %23) : (!daphne.Matrix<?x?xsi64>, ui32) -> ()
// CHECK-RECORDED: "daphne.recordProperties"(%22#0, %24) : (!daphne.Matrix<?x?xsi64>, ui32) -> ()