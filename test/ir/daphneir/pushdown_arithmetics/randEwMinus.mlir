// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 1 : index} : () -> index
    %2 = "daphne.constant"() {value = -3 : si64} : () -> si64
    %3 = "daphne.constant"() {value = -6 : si64} : () -> si64
    %4 = "daphne.constant"() {value = false} : () -> i1
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = 7 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    // CHECK: daphne.randMatrix
    // CHECK-NOT: daphne.randMatrix
    %8 = "daphne.randMatrix"(%1, %0, %3, %2, %7, %6) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: daphne.ewMinus
    %9 = "daphne.ewMinus"(%8) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    "daphne.print"(%9, %5, %4) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

