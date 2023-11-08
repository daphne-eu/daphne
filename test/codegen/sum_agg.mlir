// RUN: daphne-opt --lower-agg %s | FileCheck %s

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = 10 : index} : () -> index
    %2 = "daphne.constant"() {value = 1000000 : si64} : () -> si64
    %3 = "daphne.constant"() {value = false} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.fill"(%4, %1, %1) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    %6 = "daphne.now"() : () -> si64
    // CHECK-NOT: sumAll
    // CHECK: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // CHECK: affine.for
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: memref.load
    %7 = "daphne.sumAll"(%5) : (!daphne.Matrix<10x10xf64>) -> f64
    %8 = "daphne.now"() : () -> si64
    "daphne.print"(%7, %0, %3) : (f64, i1, i1) -> ()
    %9 = "daphne.ewSub"(%8, %6) : (si64, si64) -> si64
    %10 = "daphne.ewDiv"(%9, %2) : (si64, si64) -> si64
    "daphne.print"(%10, %0, %3) : (si64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
