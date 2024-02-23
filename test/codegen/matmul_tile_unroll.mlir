// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,3,4 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=2" %s | FileCheck %s --check-prefixes=J2
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,3,4 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=0 matmul_unroll_factor=2" %s | FileCheck %s --check-prefixes=U2
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,3,4 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=0 matmul_unroll_factor=0" %s | FileCheck %s --check-prefixes=UNROLL

module {
  func.func @main() {
    // CHECK: {{.*}}memref.alloc
    %0 = "daphne.constant"() {value = 100 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<100x100xf64>
    %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<100x100xf64>
    // CHECK: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // CHECK-NEXT: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}

    // Initialize alloced memref to 0
    // CHECK: affine.for
    // CHECK-NEXT: {{ *}}affine.for
    // CHECK-NEXT: {{ *}}affine.store

    // MatMul
    // CHECK: affine.for %arg0 = 0 to 100 step 4 {
    // CHECK-NEXT: affine.for %arg1 = 0 to 100 step 100 {
    // CHECK-NEXT: affine.for %arg2 = {{.*}} to {{.*}} step 3 {
    // CHECK-NEXT: affine.for %arg3 = {{.*}} to {{.*}} step 2 {
    // CHECK-NEXT: affine.for %arg4 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // CHECK-NOT: step
    // CHECK-SAME: {
    
    // UNROLL: affine.for %arg5 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // UNROLL-NOT: step
    // UNROLL-SAME: {
    // UNROLL-NEXT: affine.for %arg6 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // UNROLL-NOT: step
    // UNROLL-SAME: {
    // UNROLL-NEXT: {{.*}}affine.load  
    // UNROLL-NEXT: {{.*}}affine.load
    // UNROLL-NEXT: {{.*}}affine.load
    // UNROLL-NEXT: {{.*}}llvm.intr.fma
    // UNROLL-NEXT: {{.*}}affine.store

    // U2: affine.for %arg5 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // U2-NOT: step
    // U2-SAME: {
    // U2-NEXT: {{.*}}affine.load  
    // U2-NEXT: {{.*}}affine.load
    // U2-NEXT: {{.*}}affine.load
    // U2-NEXT: {{.*}}llvm.intr.fma
    // U2-NEXT: {{.*}}affine.store
    // U2-NEXT: {{.*}}affine.apply
    // U2-NEXT: {{.*}}affine.load  
    // U2-NEXT: {{.*}}affine.load
    // U2-NEXT: {{.*}}affine.load
    // U2-NEXT: {{.*}}llvm.intr.fma
    // U2-NEXT: {{.*}}affine.store

    // J2: affine.for %arg5 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // J2-NOT: step
    // J2-SAME: {
    // J2-NEXT: affine.for %arg6 = {{[a-z0-9%()#_]*}} to {{[a-z0-9%()#_]*}}
    // J2-NOT: step
    // J2-SAME: {
    // J2-NEXT: {{.*}}affine.load  
    // J2-NEXT: {{.*}}affine.load
    // J2-NEXT: {{.*}}affine.load
    // J2-NEXT: {{.*}}llvm.intr.fma
    // J2-NEXT: {{.*}}affine.store
    // J2-NEXT: {{.*}}affine.apply
    // J2-NEXT: {{.*}}affine.load  
    // J2-NEXT: {{.*}}affine.load
    // J2-NEXT: {{.*}}affine.load
    // J2-NEXT: {{.*}}llvm.intr.fma
    // J2-NEXT: {{.*}}affine.store

    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<100x100xf64>, !daphne.Matrix<100x100xf64>, i1, i1) -> !daphne.Matrix<100x100xf64>
    "daphne.return"() : () -> ()
  }
}

