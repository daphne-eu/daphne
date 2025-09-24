

// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK -dump-input=always

module {
    func.func @main() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 4 : index} : () -> index
    %2 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %4 = "daphne.constant"() {value = 3.000000e+02 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %6 = "daphne.randMatrix"(%1, %0, %3, %4, %5, %2) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
    %7 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %8 = "daphne.ewMul"(%7, %6) : (si64, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %9 = "daphne.sumAll"(%8) : (!daphne.Matrix<?x?xf64>) -> f64
    %10 = "daphne.constant"() {value = true} : () -> i1
    %11 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%9, %10, %11) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
    }
}

// CHECK: "daphne.constant"() {value = 5 : si64} : () -> si64
// CHECK: %[[MATRIX:.*]] = "daphne.randMatrix"({{.*}})
// CHECK: %[[SUM:.*]] = "daphne.sumAll"(%[[MATRIX]])
// CHECK: "daphne.ewMul"(%[[SUM]], %{{.*}}) : (!daphne.Unknown, si64) -> f64
// CHECK-NOT: "daphne.sumAll"({{.*}}ewMul)
