// RUN: daphne-opt %s | FileCheck %s --check-prefix=CHECK-INFERENCE
// RUN: daphne-opt --record-properties %s | FileCheck %s --check-prefix=CHECK-RECORDED

module {
  func.func @main() {
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %5 = "daphne.constant"() {value = 9.9999999999999995E-7 : f64} : () -> f64
    %6 = "daphne.constant"() {value = 10 : si64} : () -> si64
    %7 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %8 = "daphne.constant"() {value = 8.000000e-01 : f64} : () -> f64
    %9 = "daphne.constant"() {value = 42 : si64} : () -> si64

    // CHECK-INFERENCE: "daphne.randMatrix"({{.*}}) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>
    %10 = "daphne.randMatrix"(%0, %0, %7, %6, %8, %9) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>
    
    // CHECK-INFERENCE: "daphne.randMatrix"({{.*}}) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>
    %11 = "daphne.randMatrix"(%0, %0, %7, %6, %8, %9) : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>

    // CHECK-INFERENCE: "daphne.ewAdd"({{.*}}) : (!daphne.Matrix<10x10xsi64>, !daphne.Matrix<10x10xsi64>) -> !daphne.Matrix<10x10xsi64>
    %12 = "daphne.ewAdd"(%10, %11) : (!daphne.Matrix<10x10xsi64>, !daphne.Matrix<10x10xsi64>) -> !daphne.Matrix<10x10xsi64>
    
    %13 = "daphne.ewAdd"(%12, %5) : (!daphne.Matrix<10x10xsi64>, f64) -> !daphne.Matrix<10x10xf64>
    %14 = "daphne.ewMul"(%12, %4) : (!daphne.Matrix<10x10xsi64>, si64) -> !daphne.Matrix<10x10xsi64>
    %15 = "daphne.ewAdd"(%14, %5) : (!daphne.Matrix<10x10xsi64>, f64) -> !daphne.Matrix<10x10xf64>

    // CHECK-INFERENCE: scf.while {{.*}} -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, si64)
    %16:3 = scf.while (%arg0 = %13, %arg1 = %15, %arg2 = %7) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, si64) -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, si64) {
      %19 = "daphne.ewLe"(%arg2, %6) : (si64, si64) -> si64
      %20 = "daphne.cast"(%19) : (si64) -> i1
      scf.condition(%20) %arg0, %arg1, %arg2 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, si64
    } do {
    ^bb0(%arg0: !daphne.Matrix<10x10xf64>, %arg1: !daphne.Matrix<10x10xf64>, %arg2: si64):
      %19 = "daphne.ewLog"(%arg0, %6) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
      %20 = "daphne.ewAdd"(%arg0, %19) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
      %21 = "daphne.ewLog"(%arg1, %6) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
      %22 = "daphne.ewAdd"(%arg1, %21) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
      %23 = "daphne.ewAdd"(%arg2, %3) : (si64, si64) -> si64
      scf.yield %20, %22, %23 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, si64
    }

    %17 = "daphne.sumAll"(%16#0) : (!daphne.Matrix<10x10xf64>) -> f64
    "daphne.print"(%17, %2, %1) : (f64, i1, i1) -> ()
    %18 = "daphne.sumAll"(%16#1) : (!daphne.Matrix<10x10xf64>) -> f64
    "daphne.print"(%18, %2, %1) : (f64, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK-RECORDED: "daphne.randMatrix"({{.*}}) {daphne.value_ids = [1 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>
// CHECK-RECORDED: "daphne.recordProperties"(%10, {{.*}})
// CHECK-RECORDED: "daphne.randMatrix"({{.*}}) {daphne.value_ids = [2 : ui32]} : (index, index, si64, si64, f64, si64) -> !daphne.Matrix<10x10xsi64>
// CHECK-RECORDED: "daphne.recordProperties"(%12, {{.*}})
// CHECK-RECORDED: "daphne.ewAdd"({{.*}}) {daphne.value_ids = [3 : ui32]} : (!daphne.Matrix<10x10xsi64>, !daphne.Matrix<10x10xsi64>) -> !daphne.Matrix<10x10xsi64>
// CHECK-RECORDED: "daphne.recordProperties"(%14, {{.*}})
// CHECK-RECORDED: "daphne.ewAdd"({{.*}}) {daphne.value_ids = [4 : ui32]} : (!daphne.Matrix<10x10xsi64>, f64) -> !daphne.Matrix<10x10xf64>
// CHECK-RECORDED: "daphne.recordProperties"(%16, {{.*}})
// CHECK-RECORDED: "daphne.ewMul"({{.*}}) {daphne.value_ids = [5 : ui32]} : (!daphne.Matrix<10x10xsi64>, si64) -> !daphne.Matrix<10x10xsi64>
// CHECK-RECORDED: "daphne.recordProperties"(%18, {{.*}})
// CHECK-RECORDED: "daphne.ewAdd"({{.*}}) {daphne.value_ids = [6 : ui32]} : (!daphne.Matrix<10x10xsi64>, f64) -> !daphne.Matrix<10x10xf64>
// CHECK-RECORDED: "daphne.recordProperties"(%20, {{.*}})
// CHECK-RECORDED: scf.while {{.*}}
// CHECK-RECORDED: scf.yield %28, %30, %31
// CHECK-RECORDED: {daphne.value_ids = [7 : ui32, 8 : ui32]}
// CHECK-RECORDED: "daphne.recordProperties"(%22#1, {{.*}})
// CHECK-RECORDED: "daphne.recordProperties"(%22#0, {{.*}})
