// RUN: daphne-opt --insert-properties="properties_file_path=properties_ins_if.json" %s | FileCheck %s --check-prefix=CHECK-INSERTED -dump-input=always

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

// CHECK-INSERTED: "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-INSERTED: "daphne.randMatrix"(%0, %0, %4, %5, %6, %7) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-INSERTED: "daphne.ewAdd"(%8, %9) : (!daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>, !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-INSERTED: "daphne.ewMul"(%10, %3) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-INSERTED: %16:2 = scf.if %15 -> (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) {
// CHECK-INSERTED: "daphne.ewMul"(%11, %3) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>, si64) -> !daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>
// CHECK-INSERTED: "daphne.cast"(%21) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: "daphne.cast"(%10) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: scf.yield %22, %23 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: } else {
// CHECK-INSERTED: "daphne.cast"(%11) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: "daphne.cast"(%11) : (!daphne.Matrix<5000x5000xsi64:sp[0.95999999999999996]>) -> !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: scf.yield %21, %22 : !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>, !daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>
// CHECK-INSERTED: "daphne.cast"(%16#1) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>
// CHECK-INSERTED: "daphne.cast"(%16#0) : (!daphne.Matrix<5000x5000xsi64:sp[0.000000e+00]>) -> !daphne.Matrix<5000x5000xsi64:sp[8.000000e-01]>