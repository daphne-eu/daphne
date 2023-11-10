// RUN: daphne-opt --opt-daphne %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 2 : ui64} : () -> ui64
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 4 : ui64} : () -> ui64
    %3 = "daphne.constant"() {value = false} : () -> i1
    %4 = "daphne.constant"() {value = true} : () -> i1
    %5 = "daphne.fill"(%2, %1, %1) : (ui64, index, index) -> !daphne.Matrix<2x2xui64>
    // CHECK-NOT: daphne.ewMod
    // CHECK: daphne.ewSub
    // CHECK-NEXT: daphne.ewBitwiseAnd
    %6 = "daphne.ewMod"(%5, %0) : (!daphne.Matrix<2x2xui64>, ui64) -> !daphne.Matrix<2x2xui64>
    "daphne.print"(%6, %4, %3) : (!daphne.Matrix<2x2xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
