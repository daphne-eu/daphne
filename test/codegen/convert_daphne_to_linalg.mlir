// RUN: daphne-opt --convert-daphne-to-linalg %s | FileCheck %s

module {
  // Dense fill lowers to tensor.empty + linalg.fill (tensor form).
  func.func @fill_basic() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %m = "daphne.fill"(%c1, %c4, %c3) : (f64, index, index) -> !daphne.Matrix<4x3xf64>
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @fill_basic
    // CHECK: tensor.empty
    // CHECK: linalg.fill ins({{.*}} : f64) outs({{.*}} : tensor<4x3xf64>) -> tensor<4x3xf64>
    // CHECK-NOT: daphne.fill
  }

  // sumAll on f64 lowers to tensor.empty + linalg.fill + linalg.reduce + tensor.extract.
  func.func @sum_reduction_f64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %m = "daphne.fill"(%c1, %c4, %c3) : (f64, index, index) -> !daphne.Matrix<4x3xf64>
    %s = "daphne.sumAll"(%m) : (!daphne.Matrix<4x3xf64>) -> f64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @sum_reduction_f64
    // CHECK: tensor.empty
    // CHECK: linalg.fill ins({{.*}} : f64) outs({{.*}} : tensor<4x3xf64>) -> tensor<4x3xf64>
    // CHECK: linalg.reduce ins({{.*}} : tensor<4x3xf64>) outs({{.*}} : tensor<f64>) dimensions = [0, 1]
    // CHECK: arith.addf
    // CHECK: tensor.extract {{.*}} : tensor<f64>
    // CHECK-NOT: daphne.sumAll
  }

  // minAll on signed ints uses arith.minsi.
  func.func @min_reduction_si64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %m = "daphne.fill"(%c1, %c4, %c3) : (si64, index, index) -> !daphne.Matrix<4x3xsi64>
    %s = "daphne.minAll"(%m) : (!daphne.Matrix<4x3xsi64>) -> si64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @min_reduction_si64
    // CHECK: arith.minsi
    // CHECK-NOT: daphne.minAll
  }

  // maxAll on unsigned ints uses arith.maxui.
  func.func @max_reduction_ui64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c1 = "daphne.constant"() {value = 1 : ui64} : () -> ui64
    %m = "daphne.fill"(%c1, %c4, %c3) : (ui64, index, index) -> !daphne.Matrix<4x3xui64>
    %s = "daphne.maxAll"(%m) : (!daphne.Matrix<4x3xui64>) -> ui64
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @max_reduction_ui64
    // CHECK: arith.maxui
    // CHECK-NOT: daphne.maxAll
  }

  func.func @ewmul_mat_mat_si64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %c5 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %lhs = "daphne.fill"(%c2, %c4, %c3) : (si64, index, index) -> !daphne.Matrix<4x3xsi64>
    %rhs = "daphne.fill"(%c5, %c4, %c3) : (si64, index, index) -> !daphne.Matrix<4x3xsi64>
    %res = "daphne.ewMul"(%lhs, %rhs) : (!daphne.Matrix<4x3xsi64>, !daphne.Matrix<4x3xsi64>) -> !daphne.Matrix<4x3xsi64>
    "daphne.return"() : () -> ()
  // CHECK-LABEL: func.func @ewmul_mat_mat_si64
  // CHECK: linalg.generic
  // CHECK-SAME: outs({{.*}} : tensor<4x3xi64>)
  // CHECK: arith.muli
  // CHECK-NOT: daphne.ewMul
  }

  func.func @ewmul_mat_mat_ui64() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c2 = "daphne.constant"() {value = 2 : ui64} : () -> ui64
    %c5 = "daphne.constant"() {value = 5 : ui64} : () -> ui64
    %lhs = "daphne.fill"(%c2, %c4, %c3) : (ui64, index, index) -> !daphne.Matrix<4x3xui64>
    %rhs = "daphne.fill"(%c5, %c4, %c3) : (ui64, index, index) -> !daphne.Matrix<4x3xui64>
    %res = "daphne.ewMul"(%lhs, %rhs) : (!daphne.Matrix<4x3xui64>, !daphne.Matrix<4x3xui64>) -> !daphne.Matrix<4x3xui64>
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @ewmul_mat_mat_ui64
    // CHECK: linalg.generic
    // CHECK-SAME: outs({{.*}} : tensor<4x3xi64>)
    // CHECK: arith.muli
    // CHECK-NOT: daphne.ewMul
  }

  func.func @ewmul_mat_scalar() {
    %c4 = "daphne.constant"() {value = 4 : index} : () -> index
    %c3 = "daphne.constant"() {value = 3 : index} : () -> index
    %c2 = "daphne.constant"() {value = 2. : f64} : () -> f64
    %c5 = "daphne.constant"() {value = 5. : f64} : () -> f64
    %lhs = "daphne.fill"(%c2, %c4, %c3) : (f64, index, index) -> !daphne.Matrix<4x3xf64>
    %res = "daphne.ewMul"(%lhs, %c5) : (!daphne.Matrix<4x3xf64>, f64) -> !daphne.Matrix<4x3xf64>
    "daphne.return"() : () -> ()
    // CHECK-LABEL: func.func @ewmul_mat_scalar
    // CHECK: linalg.generic
    // CHECK-SAME: indexing_maps = [#[[ID:.*]], #[[SCALAR:.*]], #[[ID]]]
    // CHECK: arith.mulf
    // CHECK-NOT: daphne.ewMul
  }
}
