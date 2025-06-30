// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %4 = "daphne.cast"(%2) : (si64) -> index
    %5 = "daphne.cast"(%3) : (si64) -> index
    // CHECK: daphne.fill
    // CHECK-NOT: daphne.fill
    %6 = "daphne.fill"(%1, %4, %5) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: daphne.ewAdd
    %7 = "daphne.ewAdd"(%0, %6) : (si64, !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    %8 = "daphne.constant"() {value = true} : () -> i1
    %9 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%7, %8, %9) : (!daphne.Matrix<?x?xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

