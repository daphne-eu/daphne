// RUN: daphne-opt --insert-properties="properties_file_path=properties_ins_for.json" %s | FileCheck %s --check-prefix=CHECK-INSERTED -dump-input=always

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 11 : index} : () -> index
    %2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 10 : index} : () -> index
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 9.9999999999999995E-7 : f64} : () -> f64
    %7 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %9 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %10 = "daphne.constant"() {value = 42 : si64} : () -> si64
    %11 = "daphne.randMatrix"(%3, %3, %8, %7, %9, %10) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>
    %12 = "daphne.randMatrix"(%3, %3, %8, %7, %9, %10) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>
    %13 = "daphne.ewAdd"(%11, %12) : (!daphne.Matrix<10x10xsi64:sp[8.000000e-01]>, !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>
    %14 = "daphne.ewAdd"(%13, %6) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, f64) -> !daphne.Matrix<10x10xf64>
    %15 = "daphne.ewMul"(%13, %2) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>
    %16 = "daphne.ewAdd"(%15, %6) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, f64) -> !daphne.Matrix<10x10xf64>
    %17:2 = scf.for %arg0 = %0 to %1 step %0 iter_args(%arg1 = %14, %arg2 = %16) -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) {
      %20 = "daphne.cast"(%arg0) : (index) -> si64
      %21 = "daphne.ewLog"(%arg1, %7) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
      %22 = "daphne.ewAdd"(%arg1, %21) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
      %23 = "daphne.ewLog"(%arg2, %7) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
      %24 = "daphne.ewAdd"(%arg2, %23) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
      scf.yield %22, %24 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>
    }
    %18 = "daphne.sumAll"(%17#0) : (!daphne.Matrix<10x10xf64>) -> f64
    "daphne.print"(%18, %5, %4) : (f64, i1, i1) -> ()
    %19 = "daphne.sumAll"(%17#1) : (!daphne.Matrix<10x10xf64>) -> f64
    "daphne.print"(%19, %5, %4) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}


// CHECK-INSERTED: "daphne.randMatrix"(%3, %3, %8, %7, %9, %10) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>
// CHECK-INSERTED: "daphne.randMatrix"(%3, %3, %8, %7, %9, %10) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>
// CHECK-INSERTED: "daphne.ewAdd"(%11, %12) : (!daphne.Matrix<10x10xsi64:sp[8.000000e-01]>, !daphne.Matrix<10x10xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>
// CHECK-INSERTED: "daphne.ewAdd"(%13, %6) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, f64) -> !daphne.Matrix<10x10xf64>
// CHECK-INSERTED: "daphne.ewMul"(%13, %2) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>
// CHECK-INSERTED: "daphne.ewAdd"(%15, %6) : (!daphne.Matrix<10x10xsi64:sp[0.95999999999999996]>, f64) -> !daphne.Matrix<10x10xf64>
// CHECK-INSERTED: "daphne.cast"(%14) : (!daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
// CHECK-INSERTED: "daphne.cast"(%16) : (!daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
// CHECK-INSERTED: scf.for %arg0 = %0 to %1 step %0 iter_args(%arg1 = %17, %arg2 = %18) -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>)
// CHECK-INSERTED: scf.yield %26, %28 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>
// CHECK-INSERTED: "daphne.cast"(%19#1) : (!daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64:sp[1.000000e+00]>
// CHECK-INSERTED: "daphne.cast"(%19#0) : (!daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64:sp[1.000000e+00]>