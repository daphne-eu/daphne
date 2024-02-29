// RUN: daphne-opt --lower-mm="matmul_invert_loops=true" %s | FileCheck %s

module {
  func.func @main() {
    // CHECK: {{.*}}memref.alloc
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    // CHECK: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // CHECK-NEXT: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}

    // Initialize alloced memref to 0
    // CHECK: affine.for %arg0
    // CHECK-NEXT: {{ *}}affine.for %arg1
    // CHECK-NEXT: {{ *}}affine.store

    // MatMul
    // CHECK: affine.for %arg0
    // CHECK-NEXT: affine.for %arg1
    // CHECK-NEXT: affine.for %arg2
    // CHECK-NEXT: {{.*}}affine.load {{.*}}[%arg0, %arg1]
    // CHECK-NEXT: {{.*}}affine.load {{.*}}[%arg1, %arg2]
    // CHECK-NEXT: {{.*}}affine.load {{.*}}[%arg0, %arg2]
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-SAME: (f64, f64, f64) -> f64
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
    "daphne.return"() : () -> ()
  }
}