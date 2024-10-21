// RUN: daphne-opt %s | FileCheck %s --check-prefix=CHECK-PARSING
// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK-SIMPLIFIED -dump-input=always

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 5000 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 5000 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %3 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %5 = "daphne.constant"() {value = 42 : si64} : () -> si64
    %6 = "daphne.cast"(%0) : (si64) -> index
    %7 = "daphne.cast"(%1) : (si64) -> index

    // CHECK-PARSING: "daphne.randMatrix"({{.*}}) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    %8 = "daphne.randMatrix"(%6, %7, %2, %3, %4, %5) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    %9 = "daphne.constant"() {value = 5000 : si64} : () -> si64
    %10 = "daphne.constant"() {value = 5000 : si64} : () -> si64
    %11 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %12 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %13 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %14 = "daphne.constant"() {value = 42 : si64} : () -> si64
    %15 = "daphne.cast"(%9) : (si64) -> index
    %16 = "daphne.cast"(%10) : (si64) -> index

    // CHECK-PARSING: "daphne.randMatrix"({{.*}}) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
    %17 = "daphne.randMatrix"(%15, %16, %11, %12, %13, %14) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>

    // Ensure representationHint operations are present before simplification
    // CHECK-PARSING: "daphne.representationHint"(%8) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    %18 = "daphne.representationHint"(%8) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    
    // CHECK-PARSING: "daphne.representationHint"(%17) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    %19 = "daphne.representationHint"(%17) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    
    %20 = "daphne.ewAdd"(%18, %19) : (!daphne.Matrix<?x?xsi64>, !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    %21 = "daphne.ewSqrt"(%20) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xf64>

    // CHECK-PARSING: "daphne.representationHint"(%21) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64:rep[sparse]>
    %22 = "daphne.representationHint"(%21) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64:rep[sparse]>

    %23 = "daphne.sumAll"(%22) : (!daphne.Matrix<?x?xf64:rep[sparse]>) -> f64
    %24 = "daphne.constant"() {value = true} : () -> i1
    %25 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%23, %24, %25) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// Simplified output: Check after canonicalization, ensuring no representation hints remain
// CHECK-SIMPLIFIED: "daphne.randMatrix"({{.*}}) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<?x?xsi64>
// CHECK-SIMPLIFIED: "daphne.ewAdd"({{.*}}, {{.*}}) : (!daphne.Matrix<?x?xsi64>, !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
// CHECK-SIMPLIFIED: "daphne.ewSqrt"({{.*}}) : (!daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xf64:rep[sparse]>
// CHECK-SIMPLIFIED: "daphne.sumAll"
// CHECK-SIMPLIFIED-NOT: daphne.representationHint