// RUN: daphne-opt --lower-ew %s | FileCheck %s

// COM: Check conversions and presence of generic op with the right op

module {
  func.func @abs() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewAbs
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.absf
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewAbs"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewAbs
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.absf
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewAbs"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    %11 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xsi64>
    // CHECK-NOT: daphne.ewAbs
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xsi64>
    // CHECK: linalg.generic
    // CHECK: builtin.unrealized_conversion_cast %in : si64 to i64
    // CHECK-NEXT: math.absi
    // CHECK-NEXT: builtin.unrealized_conversion_cast %{{[0-9]*}} : i64 to si64
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xsi64>
    %12 = "daphne.ewAbs"(%11) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%12, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    %13 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xui64>
    // CHECK-NOT: daphne.ewAbs
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xui64>
    // CHECK: linalg.generic
    // CHECK: builtin.unrealized_conversion_cast %in : ui64 to i64
    // CHECK-NEXT: math.absi
    // CHECK-NEXT: builtin.unrealized_conversion_cast %{{[0-9]*}} : i64 to ui64
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xui64>
    %14 = "daphne.ewAbs"(%13) : (!daphne.Matrix<2x3xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%14, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @sqrt() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewSqrt
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.sqrt
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewSqrt"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewSqrt
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.sqrt
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewSqrt"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @exp() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewExp
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.exp
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewExp"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewExp
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.exp
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewExp"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @ln() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewLn
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.log
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewLn"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewLn
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.log
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewLn"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @sin() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewSin
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.sin
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewSin"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewSin
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.sin
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewSin"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @cos() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewCos
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.cos
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewCos"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewCos
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.cos
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewCos"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @floor() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewFloor
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.floor
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewFloor"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewFloor
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.floor
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewFloor"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @ceil() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewCeil
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.ceil
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewCeil"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewCeil
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.ceil
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewCeil"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @round() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 6.000000e+00 : f64} : () -> f64
    %6 = "daphne.seq"(%4, %5, %4) : (f64, f64, f64) -> !daphne.Matrix<6x1xf64:sp[1.000000e+00]>
    %7 = "daphne.reshape"(%6, %1, %0) : (!daphne.Matrix<6x1xf64:sp[1.000000e+00]>, index, index) -> !daphne.Matrix<2x3xf64:sp[1.000000e+00]>
    // CHECK-NOT: daphne.ewRound
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf64>
    // CHECK: linalg.generic
    // CHECK: math.round
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %8 = "daphne.ewRound"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%8, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    %9 = "daphne.cast"(%7) : (!daphne.Matrix<2x3xf64:sp[1.000000e+00]>) -> !daphne.Matrix<2x3xf32>
    // CHECK-NOT: daphne.ewRound
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<2x3xf32>
    // CHECK: linalg.generic
    // CHECK: math.round
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %10 = "daphne.ewRound"(%9) : (!daphne.Matrix<2x3xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%10, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}