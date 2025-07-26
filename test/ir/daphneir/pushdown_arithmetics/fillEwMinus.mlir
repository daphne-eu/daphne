// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s
module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1 : si64} : () -> si64
    // CHECK: daphne.fill
    // CHECK-NOT: daphne.fill
    %5 = "daphne.fill"(%4, %1, %0) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: daphne.ewMinus
    %6 = "daphne.ewMinus"(%5) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()

  }
}
