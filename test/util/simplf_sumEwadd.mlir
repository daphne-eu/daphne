// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK -dump-input=always

module {
  func.func @main() {
    %c1         = "daphne.constant"() {value = 4 : index} : () -> index
    %c2         = "daphne.constant"() {value = 1 : index} : () -> index
    %low        = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %high       = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %fill       = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %seed       = "daphne.constant"() {value = 42 : si64} : () -> si64
    %printFlag1 = "daphne.constant"() {value = true} : () -> i1
    %printFlag2 = "daphne.constant"() {value = false} : () -> i1

    %A = "daphne.randMatrix"(%c1, %c2, %low, %high, %fill, %seed)
       : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x1xf64>
    %B = "daphne.randMatrix"(%c1, %c2, %low, %high, %fill, %seed)
       : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x1xf64>

    %add = "daphne.ewAdd"(%A, %B)
         : (!daphne.Matrix<4x1xf64>, !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x1xf64>

    %sum = "daphne.sumAll"(%add) : (!daphne.Matrix<4x1xf64>) -> f64

    "daphne.print"(%sum, %printFlag1, %printFlag2) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK: %[[A:.*]] = "daphne.randMatrix"
// CHECK: %[[B:.*]] = "daphne.randMatrix"
// CHECK: %[[A_SUM:.*]] = "daphne.sumAll"(%[[A]]) : (!daphne.Matrix<4x1xf64>) -> !daphne.Unknown
// CHECK: %[[B_SUM:.*]] = "daphne.sumAll"(%[[B]]) : (!daphne.Matrix<4x1xf64>) -> !daphne.Unknown
// CHECK: "daphne.ewAdd"(%[[A_SUM]], %[[B_SUM]]) : (!daphne.Unknown, !daphne.Unknown) -> f64
// CHECK-NOT: "daphne.sumAll"({{.*ewAdd.*}})

