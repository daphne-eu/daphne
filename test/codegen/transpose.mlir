// RUN: daphne-opt --lower-transpose %s | FileCheck %s

// COM: Check conversions (and dimension) before and after linalg transpose op

module {
  func.func @double() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.fill"(%4, %1, %0) : (f64, index, index) -> !daphne.Matrix<3x2xf64>
    // CHECK-NOT: daphne.transpose
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<3x2xf64>
    // CHECK: linalg.transpose
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf64>
    %6 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @float() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %5 = "daphne.fill"(%4, %1, %0) : (f32, index, index) -> !daphne.Matrix<3x2xf32>
    // CHECK-NOT: daphne.transpose
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<3x2xf32>
    // CHECK: linalg.transpose
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xf32>
    %6 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xf32>) -> !daphne.Matrix<2x3xf32>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<2x3xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @signedIntegers() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %5 = "daphne.fill"(%4, %1, %0) : (si64, index, index) -> !daphne.Matrix<3x2xsi64>
    // CHECK-NOT: daphne.transpose
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<3x2xsi64>
    // CHECK: linalg.transpose
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xsi64>
    %6 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<2x3xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @unsignedIntegers() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1 : ui64} : () -> ui64
    %5 = "daphne.fill"(%4, %1, %0) : (ui64, index, index) -> !daphne.Matrix<3x2xui64>
    // CHECK-NOT: daphne.transpose
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<3x2xui64>
    // CHECK: linalg.transpose
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<2x3xui64>
    %6 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xui64>) -> !daphne.Matrix<2x3xui64>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<2x3xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}