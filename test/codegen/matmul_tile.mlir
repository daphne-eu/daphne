// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2,2,2 matmul_use_fixed_tile_sizes=true" %s | FileCheck %s --check-prefixes=T5
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2,2 matmul_use_fixed_tile_sizes=true" %s | FileCheck %s --check-prefixes=T4
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2 matmul_use_fixed_tile_sizes=true" %s | FileCheck %s --check-prefixes=T3

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
    // T5: affine.for %arg0 = 0 to 10 step 2 {
    // T5-NEXT: affine.for %arg1 = 0 to 10 step 2 {
    // T5-NEXT: affine.for %arg2 = 0 to 10 step 2 {
    // T5-NEXT: affine.for %arg3 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}} step 2 {
    // T5-NEXT: affine.for %arg4 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}} step 2 {
    // T5-NEXT: affine.for %arg5 = {{[a-z0-9%()#_]*}}
    // T5-NOT: step
    // T5-SAME: {

    // T4: affine.for %arg0 = 0 to 10 step 2 {
    // T4-NEXT: affine.for %arg1 = 0 to 10 step 2 {
    // T4-NEXT: affine.for %arg2 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}} step 2 {
    // T4-NEXT: affine.for %arg3 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}} step 2 {
    // T4-NEXT: affine.for %arg4 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // T4-NOT: step
    // T4-SAME: {

    // T3: affine.for %arg0 = 0 to 10 step 2 {
    // T3-NEXT: affine.for %arg1 = 0 to 10 step 10 {
    // T3-NEXT: affine.for %arg2 = {{.*}} to {{.*}} step 2 {
    // T3-NEXT: affine.for %arg3 = {{.*}} to {{.*}} step 2 {
    // T3-NEXT: affine.for %arg4 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // T3-NOT: step
    // T3-SAME: {
    
    // CHECK-NEXT: {{.*}}affine.load  
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-NEXT: {{.*}}affine.store
    // CHECK-NEXT: {{.*}}affine.apply
    // CHECK-NEXT: {{.*}}affine.load  
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-NEXT: {{.*}}affine.store
    // CHECK-NEXT: {{.*}}affine.apply
    // CHECK-NEXT: {{.*}}affine.load  
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-NEXT: {{.*}}affine.store
    // CHECK-NEXT: {{.*}}affine.apply
    // CHECK-NEXT: {{.*}}affine.load  
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
    "daphne.return"() : () -> ()
  }
}

// Non-square matrix multiplications are also lowered but not tiled
module {
  func.func @main3() {
    // CHECK: {{.*}}memref.alloc
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x5xf64>
    %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<5x10xf64>
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
    // CHECK-SAME: (f32, f32, f32) -> f32
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x5xf64>, !daphne.Matrix<5x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
    "daphne.return"() : () -> ()
  }
}