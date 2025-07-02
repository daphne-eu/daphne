// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 1 : si64} : () -> si64
    // CHECK: daphne.ewPow
    %6 = "daphne.ewMinus"(%5) : (si64) -> si64
    %7 = "daphne.cast"(%0) : (si64) -> index
    %8 = "daphne.cast"(%1) : (si64) -> index
    // CHECK: daphne.randMatrix
    // CHECK-NOT: daphne.randMatrix
    %9 = "daphne.randMatrix"(%7, %8, %2, %3, %4, %6) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    %10 = "daphne.constant"() {value = 2 : si64} : () -> si64
    // CHECK-NOT: daphne.ewPow
    %11 = "daphne.ewPow"(%9, %10) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    %12 = "daphne.constant"() {value = true} : () -> i1
    %13 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%11, %12, %13) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
