// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK -dump-input=always

module {
  func.func @main() {
    %rows = "daphne.constant"() {value = 2 : index} : () -> index
    %cols = "daphne.constant"() {value = 2 : index} : () -> index
    %low  = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %high = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %fill = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %seed = "daphne.constant"() {value = 42 : si64} : () -> si64

    %lhs = "daphne.randMatrix"(%rows, %cols, %low, %high, %fill, %seed)
         : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x2xf64>
    %rhs = "daphne.randMatrix"(%rows, %cols, %low, %high, %fill, %seed)
         : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x2xf64>

    %tX = "daphne.constant"() {value = false} : () -> i1
    %tY = "daphne.constant"() {value = false} : () -> i1

    %mm = "daphne.matMul"(%lhs, %rhs, %tX, %tY)
        : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>

    %diag = "daphne.diagVector"(%mm)
          : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x1xf64>

    %sum = "daphne.sumAll"(%diag)
         : (!daphne.Matrix<2x1xf64>) -> f64

    "daphne.return"() : () -> ()
  }
}


// CHECK: %[[LHS:.*]] = "daphne.randMatrix"({{.*}}) : {{.*}} -> !daphne.Matrix<2x2xf64>
// CHECK: %[[RHS:.*]] = "daphne.randMatrix"({{.*}}) : {{.*}} -> !daphne.Matrix<2x2xf64>
// CHECK: %[[TRHS:.*]] = "daphne.transpose"(%[[RHS]]) : (!daphne.Matrix<2x2xf64>) -> !daphne.Unknown
// CHECK: %[[EWMUL:.*]] = "daphne.ewMul"(%[[LHS]], %[[TRHS]]) : (!daphne.Matrix<2x2xf64>, !daphne.Unknown) -> !daphne.Unknown
// CHECK: "daphne.sumAll"(%[[EWMUL]]) : (!daphne.Unknown) -> f64
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagVector

