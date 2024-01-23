// RUN: daphne-opt --lower-ew %s | FileCheck %s

func.func @add() {
  %0 = "daphne.constant"() {value = 2 : index} : () -> index
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = true} : () -> i1
  %3 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  // CHECK-NOT: daphne.ewAdd
  // CHECK: arith.addf
  %5 = "daphne.ewAdd"(%4, %4) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  "daphne.print"(%5, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @sub() {
  %0 = "daphne.constant"() {value = 2 : index} : () -> index
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = true} : () -> i1
  %3 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  // CHECK-NOT: daphne.ewSub
  // CHECK: arith.subf
  %5 = "daphne.ewSub"(%4, %4) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  "daphne.print"(%5, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @mul() {
  %0 = "daphne.constant"() {value = 2 : index} : () -> index
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = true} : () -> i1
  %3 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  // CHECK-NOT: daphne.ewMul
  // CHECK: arith.mulf
  %5 = "daphne.ewMul"(%4, %4) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  "daphne.print"(%5, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @div() {
  %0 = "daphne.constant"() {value = 2 : index} : () -> index
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = true} : () -> i1
  %3 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  // CHECK-NOT: daphne.ewDiv
  // CHECK: arith.divf
  %5 = "daphne.ewDiv"(%4, %4) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  "daphne.print"(%5, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @sqrt() {
  %0 = "daphne.constant"() {value = 0 : index} : () -> si64
  %1 = "daphne.constant"() {value = 1 : index} : () -> si64
  %2 = "daphne.constant"() {value = 2 : index} : () -> index
  %3 = "daphne.constant"() {value = false} : () -> i1
  %4 = "daphne.constant"() {value = true} : () -> i1
  %5 = "daphne.constant"() {value = 4 : si64} : () -> si64
  %6 = "daphne.fill"(%5, %2, %2) : (si64, index, index) -> !daphne.Matrix<2x2xsi64>
  %7 = "daphne.sliceRow"(%6, %0, %1) : (!daphne.Matrix<2x2xsi64>, si64, si64) -> !daphne.Matrix<?x?xsi64>
  %8 = "daphne.sliceCol"(%7, %0, %1) : (!daphne.Matrix<?x?xsi64>, si64, si64) -> !daphne.Matrix<?x?xsi64>
  %9 = "daphne.sliceRow"(%6, %0, %1) : (!daphne.Matrix<2x2xsi64>, si64, si64) -> !daphne.Matrix<?x?xsi64>
  %10 = "daphne.sliceCol"(%9, %0, %1) : (!daphne.Matrix<?x?xsi64>, si64, si64) -> !daphne.Matrix<?x?xsi64>
  %11 = "daphne.cast"(%10) : (!daphne.Matrix<?x?xsi64>) -> si64
  %12 = "daphne.cast"(%11) : (si64) -> f64
  // CHECK-NOT: daphne.ewSqrt
  // CHECK: math.sqrt
  %13 = "daphne.ewSqrt"(%12) : (f64) -> f64
  "daphne.print"(%13, %4, %3) : (f64, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @abs() {
  %0 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %3 = "daphne.constant"() {value = false} : () -> i1
  %4 = "daphne.constant"() {value = true} : () -> i1
  // CHECK-NOT: daphne.ewAbs
  // CHECK: math.absf
  %12 = "daphne.ewAbs"(%0) : (f64) -> f64
  "daphne.print"(%12, %4, %3) : (f64, i1, i1) -> ()
  "daphne.return"() : () -> ()
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
