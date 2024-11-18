// RUN: daphne-opt --lower-ew %s | FileCheck %s

// COM: Check whether op has been correctly replaced by generic op and corresponding binary op.
// COM: Conversions from/to MemRef/DenseMatrix are moved by canonicalizer and are hence not checked here.

module {
  func.func @add() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewAdd
    // CHECK: linalg.generic
    // CHECK: arith.addf
    %8 = "daphne.ewAdd"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewAdd
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %11 = "daphne.ewAdd"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewAdd
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %14 = "daphne.ewAdd"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @sub() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewSub
    // CHECK: linalg.generic
    // CHECK: arith.subf
    %8 = "daphne.ewSub"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewSub
    // CHECK: linalg.generic
    // CHECK: arith.subi
    %11 = "daphne.ewSub"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewSub
    // CHECK: linalg.generic
    // CHECK: arith.subi
    %14 = "daphne.ewSub"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @mul() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewMul
    // CHECK: linalg.generic
    // CHECK: arith.mulf
    %8 = "daphne.ewMul"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewMul
    // CHECK: linalg.generic
    // CHECK: arith.muli
    %11 = "daphne.ewMul"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewMul
    // CHECK: linalg.generic
    // CHECK: arith.muli
    %14 = "daphne.ewMul"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @div() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewDiv
    // CHECK: linalg.generic
    // CHECK: arith.divf
    %8 = "daphne.ewDiv"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewDiv
    // CHECK: linalg.generic
    // CHECK: arith.divsi
    %11 = "daphne.ewDiv"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewDiv
    // CHECK: linalg.generic
    // CHECK: arith.divui
    %14 = "daphne.ewDiv"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

func.func @pow() {
  %0 = "daphne.constant"() {value = 0 : index} : () -> si64
  %1 = "daphne.constant"() {value = 1 : index} : () -> si64
  %2 = "daphne.constant"() {value = 2 : index} : () -> index
  %3 = "daphne.constant"() {value = false} : () -> i1
  %4 = "daphne.constant"() {value = true} : () -> i1
  %5 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %6 = "daphne.fill"(%5, %2, %2) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  %7 = "daphne.sliceRow"(%6, %0, %1) : (!daphne.Matrix<2x2xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
  %8 = "daphne.sliceCol"(%7, %0, %1) : (!daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
  %9 = "daphne.sliceRow"(%6, %0, %1) : (!daphne.Matrix<2x2xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
  %10 = "daphne.sliceCol"(%9, %0, %1) : (!daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
  %11 = "daphne.cast"(%10) : (!daphne.Matrix<?x?xf64>) -> f64
  // CHECK-NOT: daphne.ewPow
  // CHECK: math.powf
  %12 = "daphne.ewPow"(%11, %11) : (f64, f64) -> f64
  "daphne.print"(%12, %4, %3) : (f64, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

module {
  func.func @max() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewMax
    // CHECK: linalg.generic
    // CHECK: arith.maxf
    %8 = "daphne.ewMax"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewMax
    // CHECK: linalg.generic
    // CHECK: arith.maxsi
    %11 = "daphne.ewMax"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewMax
    // CHECK: linalg.generic
    // CHECK: arith.maxui
    %14 = "daphne.ewMax"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @min() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewMin
    // CHECK: linalg.generic
    // CHECK: arith.minf
    %8 = "daphne.ewMin"(%7, %7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>, !daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewMin
    // CHECK: linalg.generic
    // CHECK: arith.minsi
    %11 = "daphne.ewMin"(%9, %10) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %12 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewMin
    // CHECK: linalg.generic
    // CHECK: arith.minui
    %14 = "daphne.ewMin"(%12, %13) : (!daphne.Matrix<2x3xui64>, !daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}