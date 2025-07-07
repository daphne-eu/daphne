// RUN: daphne-opt --record-properties %s | FileCheck %s --check-prefix=CHECK-RECORDED -dump-input=always

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 5000 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %5 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %6 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %7 = "daphne.constant"() {value = 42 : si64} : () -> si64
    %8 = "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
    %9 = "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
    %10 = "daphne.ewAdd"(%8, %9) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
    %11 = "daphne.ewMul"(%10, %3) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
    %12 = "daphne.sumAll"(%10) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> si64
    %13 = "daphne.sumAll"(%11) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> si64
    %14 = "daphne.ewGt"(%12, %13) : (si64, si64) -> si64
    %15 = "daphne.cast"(%14) : (si64) -> i1
    %16:2 = scf.if %15 -> (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) {
      %19 = "daphne.ewMul"(%11, %3) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
      %20 = "daphne.cast"(%19) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
      %21 = "daphne.cast"(%10) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
      scf.yield %20, %21 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
    } else {
      %19 = "daphne.cast"(%11) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
      %20 = "daphne.cast"(%11) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
      scf.yield %19, %20 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
    }
    %17 = "daphne.sumAll"(%16#1) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) -> si64
    "daphne.print"(%17, %2, %1) : (si64, i1, i1) -> ()
    %18 = "daphne.sumAll"(%16#0) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) -> si64
    "daphne.print"(%18, %2, %1) : (si64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK-RECORDED: "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) {daphne.value_ids = [1 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-RECORDED: "daphne.recordProperties"(%8, %9) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, ui32) -> ()
// CHECK-RECORDED: "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) {daphne.value_ids = [2 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-RECORDED: "daphne.recordProperties"(%10, %11) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, ui32) -> ()
// CHECK-RECORDED: "daphne.ewAdd"(%8, %10) {daphne.value_ids = [3 : ui32]} : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-RECORDED: "daphne.recordProperties"(%12, %13) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, ui32) -> ()
// CHECK_RECORDED: "daphne.ewMul"(%12, %3) {daphne.value_ids = [4 : ui32]} : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-RECORDED: "daphne.recordProperties"(%14, %15) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, ui32) -> ()
// CHECK-RECORDED: scf.if %19 -> (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) {
// CHECK-RECORDED: "daphne.ewMul"(%14, %3) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-RECORDED: scf.yield %26, %27 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-RECORDED: } else {
// CHECK-RECORDED: "daphne.cast"(%14) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-RECORDED: "daphne.cast"(%14) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-RECORDED: scf.yield %25, %26 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-RECORDED: } {daphne.value_ids = [5 : ui32, 6 : ui32]}
// CHECK-RECORDED: "daphne.recordProperties"(%20#1, %21) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, ui32) -> ()
// CHECK-RECORDED: "daphne.recordProperties"(%20#0, %22) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, ui32) -> ()