// RUN: daphne-opt --lower-agg-dim %s | FileCheck %s

// COM: Check whether op has been correctly replaced by generic op and corresponding agg op.
// COM: Conversions from/to MemRef/DenseMatrix are moved by canonicalizer and are hence not checked here.

module {
  func.func @rowSum() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94383135183824 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.sumRow
    // CHECK: linalg.generic
    // CHECK: arith.addf
    %7 = "daphne.sumRow"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x1xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x1xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.sumRow
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %9 = "daphne.sumRow"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x1xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<2x1xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.sumRow
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %11 = "daphne.sumRow"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x1xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x1xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @colSum() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94362996892656 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.sumCol
    // CHECK: linalg.generic
    // CHECK: arith.addf
    %7 = "daphne.sumCol"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<1x3xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<1x3xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.sumCol
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %9 = "daphne.sumCol"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<1x3xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<1x3xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.sumCol
    // CHECK: linalg.generic
    // CHECK: arith.addi
    %11 = "daphne.sumCol"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<1x3xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<1x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @rowMin() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94588503285472 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.minRow
    // CHECK: linalg.generic
    // CHECK: arith.minf
    %7 = "daphne.minRow"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x1xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x1xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.minRow
    // CHECK: linalg.generic
    // CHECK: arith.minsi
    %9 = "daphne.minRow"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x1xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<2x1xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.minRow
    // CHECK: linalg.generic
    // CHECK: arith.minui
    %11 = "daphne.minRow"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x1xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x1xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @colMin() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94300152631008 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.minCol
    // CHECK: linalg.generic
    // CHECK: arith.minf
    %7 = "daphne.minCol"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<1x3xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<1x3xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.minCol
    // CHECK: linalg.generic
    // CHECK: arith.minsi
    %9 = "daphne.minCol"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<1x3xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<1x3xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.minCol
    // CHECK: linalg.generic
    // CHECK: arith.minui
    %11 = "daphne.minCol"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<1x3xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<1x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @rowMax() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94739586083856 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.maxRow
    // CHECK: linalg.generic
    // CHECK: arith.maxf
    %7 = "daphne.maxRow"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x1xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x1xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.maxRow
    // CHECK: linalg.generic
    // CHECK: arith.maxsi
    %9 = "daphne.maxRow"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x1xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<2x1xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.maxRow
    // CHECK: linalg.generic
    // CHECK: arith.maxui
    %11 = "daphne.maxRow"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x1xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x1xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @colMax() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94443432202464 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.maxCol
    // CHECK: linalg.generic
    // CHECK: arith.maxf
    %7 = "daphne.maxCol"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<1x3xf64>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<1x3xf64>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.maxCol
    // CHECK: linalg.generic
    // CHECK: arith.maxsi
    %9 = "daphne.maxCol"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<1x3xsi64>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<1x3xsi64>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.maxCol
    // CHECK: linalg.generic
    // CHECK: arith.maxui
    %11 = "daphne.maxCol"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<1x3xui64>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<1x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @rowIdxMin() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94602475382496 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.idxminRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpf ole
    // CHECK: arith.select
    %7 = "daphne.idxminRow"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.idxminRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi sle
    // CHECK: arith.select
    %9 = "daphne.idxminRow"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.idxminRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi ule
    // CHECK: arith.select
    %11 = "daphne.idxminRow"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @colIdxMin() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94805958167520 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.idxminCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpf ole
    // CHECK: arith.select
    %7 = "daphne.idxminCol"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.idxminCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi sle
    // CHECK: arith.select
    %9 = "daphne.idxminCol"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.idxminCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi ule
    // CHECK: arith.select
    %11 = "daphne.idxminCol"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @rowIdxMax() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94221441505504 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    // CHECK-NOT: daphne.idxmaxRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpf oge
    // CHECK: arith.select
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    %7 = "daphne.idxmaxRow"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.idxmaxRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi sge
    // CHECK: arith.select
    %9 = "daphne.idxmaxRow"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.idxmaxRow
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi uge
    // CHECK: arith.select
    %11 = "daphne.idxmaxRow"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x1xindex>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<2x1xindex>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @colIdxMax() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 94198175002784 : ui64} : () -> ui64
    %5 = "daphne.matrixConstant"(%4) : (ui64) -> !daphne.Matrix<6x1xf64>
    %6 = "daphne.reshape"(%5, %0, %1) : (!daphne.Matrix<6x1xf64>, index, index) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: daphne.idxmaxCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpf oge
    // CHECK: arith.select
    %7 = "daphne.idxmaxCol"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    %8 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.idxmaxCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi sge
    // CHECK: arith.select
    %9 = "daphne.idxmaxCol"(%8) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%9, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    %10 = "daphne.cast"(%6) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.idxmaxCol
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: arith.cmpi uge
    // CHECK: arith.select
    %11 = "daphne.idxmaxCol"(%10) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<1x3xindex>
    "daphne.print"(%11, %3, %2) : (!daphne.Matrix<1x3xindex>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}