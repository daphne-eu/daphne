// RUN: daphne-opt --convert-daphne-to-linalg %s | FileCheck %s

module {
  // Dense fill lowers to memref.alloc + linalg.fill.
  func.func @fill_basic() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %m = "daphne.fill"(%c1, %c4, %c3) : (f64, index, index) -> !daphne.Matrix<4x3xf64>
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @fill_basic
    // CHECK: memref.alloc
    // CHECK: linalg.fill
    // CHECK-NOT: daphne.fill
  }

  // sumAll on f64 lowers to bufferization.to_tensor + linalg.reduce + tensor.extract.
  func.func @sum_f64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %m = "daphne.fill"(%c1, %c4, %c3) : (f64, index, index) -> !daphne.Matrix<4x3xf64>
    %s = "daphne.sumAll"(%m) : (!daphne.Matrix<4x3xf64>) -> f64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @sum_f64
    // CHECK: bufferization.to_tensor {{.*}} restrict : memref<4x3xf64> to tensor<4x3xf64>
    // CHECK: linalg.reduce
    // CHECK: arith.addf
    // CHECK: tensor.extract
    // CHECK-NOT: daphne.sumAll
  }

  // minAll on signed ints uses arith.minsi.
  func.func @min_si64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %m = "daphne.fill"(%c1, %c4, %c3) : (si64, index, index) -> !daphne.Matrix<4x3xsi64>
    %s = "daphne.minAll"(%m) : (!daphne.Matrix<4x3xsi64>) -> si64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @min_si64
    // CHECK: arith.minsi
    // CHECK-NOT: daphne.minAll
  }

  // maxAll on unsigned ints uses arith.maxui.
  func.func @max_ui64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1 : ui64} : () -> ui64
    %m = "daphne.fill"(%c1, %c4, %c3) : (ui64, index, index) -> !daphne.Matrix<4x3xui64>
    %s = "daphne.maxAll"(%m) : (!daphne.Matrix<4x3xui64>) -> ui64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @max_ui64
    // CHECK: arith.maxui
    // CHECK-NOT: daphne.maxAll
  }
}
