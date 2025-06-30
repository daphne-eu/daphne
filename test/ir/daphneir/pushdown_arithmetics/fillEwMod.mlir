// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %3 = "daphne.cast"(%1) : (si64) -> index
    %4 = "daphne.cast"(%2) : (si64) -> index
    // CHECK: daphne.fill
    // CHECK-NOT: daphne.fill
    %5 = "daphne.fill"(%0, %3, %4) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
    %6 = "daphne.constant"() {value = 5 : si64} : () -> si64
    // CHECK-NOT: daphne.ewMod
    %7 = "daphne.ewMod"(%5, %6) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    %8 = "daphne.constant"() {value = true} : () -> i1
    %9 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%7, %8, %9) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
