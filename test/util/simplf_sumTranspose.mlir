// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK -dump-input=always

module {
  func.func @main() {
    %c1 = "daphne.constant"() {value = 2 : index} : () -> index
    %c2 = "daphne.constant"() {value = 3 : index} : () -> index
    %low = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %high = "daphne.constant"() {value = 10.0 : f64} : () -> f64
    %fill = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %seed = "daphne.constant"() {value = 42 : si64} : () -> si64
    %true = "daphne.constant"() {value = true} : () -> i1
    %false = "daphne.constant"() {value = false} : () -> i1

    %A = "daphne.randMatrix"(%c1, %c2, %low, %high, %fill, %seed)
       : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<2x3xf64>

    %AT = "daphne.transpose"(%A)
         : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<3x2xf64>

    %s = "daphne.sumAll"(%AT)
       : (!daphne.Matrix<3x2xf64>) -> f64

    "daphne.print"(%s, %true, %false) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK: %[[A:.*]] = "daphne.randMatrix"
// CHECK-NOT: "daphne.transpose"
// CHECK: "daphne.sumAll"(%[[A]]) : (!daphne.Matrix<2x3xf64>) -> f64

