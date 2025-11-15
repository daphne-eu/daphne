// RUN: daphne-opt --lower-agg %s | FileCheck %s

// COM: Check whether op has been correctly replaced by generic op and corresponding agg op.
// COM: Conversions from/to MemRef/DenseMatrix are moved by canonicalizer and are hence not checked here.

module {
  func.func @sum() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94652999131312 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.sumAll
    // CHECK: linalg.generic
    // CHECK: arith.addf
    %7 = "daphne.sumAll"(%6) : (!daphne.Matrix<2x3xf64>) -> f64
    "daphne.print"(%7, %3, %2) : (f64, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.sumAll
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %9 = "daphne.sumAll"(%8) : (!daphne.Matrix<2x3xsi64>) -> si64
    "daphne.print"(%9, %3, %2) : (si64, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.sumAll
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %11 = "daphne.sumAll"(%10) : (!daphne.Matrix<2x3xui64>) -> ui64
    "daphne.print"(%11, %3, %2) : (ui64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @aggMin() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94845281352640 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.minAll
    // CHECK: linalg.generic
    // CHECK: arith.minimumf
    %7 = "daphne.minAll"(%6) : (!daphne.Matrix<2x3xf64>) -> f64
    "daphne.print"(%7, %3, %2) : (f64, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.minAll
    // CHECK: linalg.generic
    // CHECK: arith.minsi
    %9 = "daphne.minAll"(%8) : (!daphne.Matrix<2x3xsi64>) -> si64
    "daphne.print"(%9, %3, %2) : (si64, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.minAll
    // CHECK: linalg.generic
    // CHECK: arith.minui
    %11 = "daphne.minAll"(%10) : (!daphne.Matrix<2x3xui64>) -> ui64
    "daphne.print"(%11, %3, %2) : (ui64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @aggMax() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94592435720432 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.maxAll
    // CHECK: linalg.generic
    // CHECK: arith.maximumf
    %7 = "daphne.maxAll"(%6) : (!daphne.Matrix<2x3xf64>) -> f64
    "daphne.print"(%7, %3, %2) : (f64, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.maxAll
    // CHECK: linalg.generic
    // CHECK: arith.maxsi
    %9 = "daphne.maxAll"(%8) : (!daphne.Matrix<2x3xsi64>) -> si64
    "daphne.print"(%9, %3, %2) : (si64, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.maxAll
    // CHECK: linalg.generic
    // CHECK: arith.maxui
    %11 = "daphne.maxAll"(%10) : (!daphne.Matrix<2x3xui64>) -> ui64
    "daphne.print"(%11, %3, %2) : (ui64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
