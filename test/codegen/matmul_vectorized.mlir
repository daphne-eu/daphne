// RUN: daphne-opt --lower-mm="matmul_vec_size_bits=128" %s | FileCheck %s --check-prefix=ONE28
// RUN: daphne-opt --lower-mm="matmul_vec_size_bits=64" %s | FileCheck %s --check-prefix=SIX4
// RUN: daphne-opt --lower-mm="matmul_vec_size_bits=32" %s | FileCheck %s --check-prefix=THREE2

// double
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
    // CHECK: affine.for
    // CHECK-NEXT: {{ *}}affine.for
    // CHECK-NEXT: {{ *}}affine.store

    // MatMul
    // CHECK: affine.for
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}vector.splat
    // CHECK-NEXT: {{.*}}affine.vector_load
    // CHECK-NEXT: {{.*}}affine.vector_load
    // SIX4: {{.*}}vector.fma
    // SIX4-SAME: vector<1xf64>
    // SIX4-NEXT: {{.*}}affine.vector_store

    // THREE2: {{.*}}vector.fma
    // THREE2-SAME: vector<1xf64>
    // THREE2-NEXT: {{.*}}affine.vector_store

    // COM: If enabled always at least one element is packed into a vector.
    // ONE: {{.*}}vector.fma
    // ONE-SAME: vector<1xf64>
    // ONE-NEXT: {{.*}}affine.vector_store

    // ONE28: %13 = vector.fma
    // ONE28-SAME: vector<2xf64>
    // ONE28-NEXT: {{.*}}affine.vector_store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
    "daphne.return"() : () -> ()
  }
}

// Value type single
// See 128 bit vector, that vectorization does not happen, if vector size does not divide the matrix size.
module {
  func.func @main() {
    // CHECK: {{.*}}memref.alloc
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f32} : () -> f32
    %3 = "daphne.constant"() {value = 5.000000e+00 : f32} : () -> f32
    %4 = "daphne.fill"(%3, %0, %0) : (f32, index, index) -> !daphne.Matrix<10x10xf32>
    %5 = "daphne.fill"(%2, %0, %0) : (f32, index, index) -> !daphne.Matrix<10x10xf32>
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
    // CHECK-NEXT: {{.*}}vector.splat
    // CHECK-NEXT: {{.*}}affine.vector_load
    // CHECK-NEXT: {{.*}}affine.vector_load
    // SIX4: {{.*}}vector.fma
    // SIX4-SAME: vector<2xf32>
    // SIX4-NEXT: {{.*}}affine.vector_store

    // THREE2: {{.*}}vector.fma
    // THREE2-SAME: vector<1xf32>
    // THREE2-NEXT: {{.*}}affine.vector_store
    
    // COM: Vec size bits 128 gives f32 vector with 4 elements. 4 does not divide 10, so the matmul is not vectorized.
    // ONE28: {{.*}}affine.load
    // ONE28-NEXT: {{.*}}affine.load
    // ONE28-NEXT: {{.*}}affine.load
    // ONE28-NEXT: {{.*}}llvm.intr.fma
    // ONE28-SAME: (f32, f32, f32) -> f32
    // ONE28-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf32>, !daphne.Matrix<10x10xf32>, i1, i1) -> !daphne.Matrix<10x10xf32>
    "daphne.return"() : () -> ()
  }
}