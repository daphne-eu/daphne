// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = -1 : si64} : () -> si64
    %2 = "daphne.constant"() {value = -2 : si64} : () -> si64
    %3 = "daphne.constant"() {value = false} : () -> i1
    %4 = "daphne.constant"() {value = true} : () -> i1
    %5 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %6 = "daphne.constant"() {value = 3 : si64} : () -> si64
    // CHECK: daphne.randMatrix
    // CHECK-NOT: daphne.randMatrix
    %7 = "daphne.randMatrix"(%0, %0, %2, %6, %5, %1) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: daphne.ewExp
    %8 = "daphne.ewExp"(%7) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%8, %4, %3) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
