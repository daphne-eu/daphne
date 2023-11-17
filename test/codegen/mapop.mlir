// RUN: daphne-opt --lower-map --inline %s | FileCheck %s

module {
  func.func @"increment-1-1"(%arg0: f64) -> f64 {
    %0 = "daphne.ewExp"(%arg0) : (f64) -> f64
    "daphne.return"(%0) : (f64) -> ()
  }
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 93985655361872 : ui64} : () -> ui64
    %4 = "daphne.matrixConstant"(%3) : (ui64) -> !daphne.Matrix<?x?xf64>
    %5 = "daphne.reshape"(%4, %0, %0) : (!daphne.Matrix<?x?xf64>, index, index) -> !daphne.Matrix<2x2xf64>
    // CHECK-NOT: daphne.map
    // CHECK: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // CHECK: affine.for
    // CHECK-NEXT: affine.for
    // CHECK-NOT: func.call
    // CHECK: affine.load
    // CHECK-NEXT: daphne.ewExp
    %6 = "daphne.map"(%5) {func = "increment-1-1"} : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
    "daphne.print"(%6, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
