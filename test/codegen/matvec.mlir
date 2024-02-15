// RUN: daphne-opt --lower-mm %s | FileCheck %s 
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2,2,2 matmul_use_fixed_tile_sizes=true" %s | FileCheck %s


// Matrix Vector products are lowered but not tiled
module {
  func.func @main() {
    // CHECK: {{.*}}memref.alloc
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x1xf64>
    // CHECK: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // CHECK-NEXT: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}

    // Initialize alloced memref to 0
    // CHECK: affine.for
    // CHECK-NEXT: {{ *}}affine.for
    // CHECK-NEXT: {{ *}}affine.store

    // MatMul
    // CHECK: affine.for
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x1xf64>, i1, i1) -> !daphne.Matrix<10x1xf64>
    "daphne.return"() : () -> ()
  }
}