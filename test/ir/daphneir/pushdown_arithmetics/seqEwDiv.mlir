// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
    func.func @main() {
    %0 = "daphne.constant"() {value = false} : () -> i1
    %1 = "daphne.constant"() {value = true} : () -> i1
    %2 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 2 : si64} : () -> si64
    // CHECK: daphne.seq
    // CHECK-NOT: daphne.seq
    %5 = "daphne.seq"(%2, %3, %4) : (si64, si64, si64) -> !daphne.Matrix<?x?xsi64>
    // CHECK-NOT: ewDiv
    %6 = "daphne.ewDiv"(%5, %4) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.print"(%6, %1, %0) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }

}

