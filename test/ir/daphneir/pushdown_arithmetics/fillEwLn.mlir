// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = -3 : si64} : () -> si64
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    // CHECK: daphne.fill
    // CHECK-NOT: daphne.fill
    %4 = "daphne.fill"(%1, %0, %0) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: daphne.ewLn
    %5 = "daphne.ewLn"(%4) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    "daphne.print"(%5, %3, %2) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
