// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2,2,2 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=4" --affine-loop-normalize %s | FileCheck %s --check-prefixes=NT5,NT,NTL
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2,2 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=4" --affine-loop-normalize %s | FileCheck %s --check-prefixes=NT4,NT,NTL
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2,2 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=4" --affine-loop-normalize %s | FileCheck %s --check-prefixes=NT3,NT,NTL
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2,2 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=4" --affine-loop-normalize %s | FileCheck %s --check-prefixes=NT2,NT,NTS
// RUN: daphne-opt --lower-mm="matmul_tile=true matmul_fixed_tile_sizes=2 matmul_use_fixed_tile_sizes=true matmul_unroll_jam_factor=4" --affine-loop-normalize %s | FileCheck %s --check-prefixes=NT1,NT,NTS

// Non specified tile sizes are filled in by the matrix size.
// The innermost loop has many instruction, when one of the unroll factors > 1.
// Here we test normalized loops with step size 1 in all loops.
module {
  func.func @main() {
    // NT: {{.*}}memref.alloc
      %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
    %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<200x200xf64>
    %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<200x200xf64>
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<200x200xf64>, !daphne.Matrix<200x200xf64>, i1, i1) -> !daphne.Matrix<200x200xf64>
    // NT: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}
    // NT-NEXT: {{.*}}"daphne.convertDenseMatrixToMemRef"{{.*}}

    // Initialize alloced memref to 0
    // NT: affine.for
    // NT-NEXT: {{ *}}affine.for
    // NT-NEXT: {{ *}}affine.store

    // NT5: affine.for %arg0 = 0 to 100 {
    // NT5-NEXT: {{.*}}affine.apply
    // NT5-NEXT: affine.for %arg1 = 0 to 100 {
    // NT5-NEXT: {{.*}}affine.apply
    // NT5-NEXT: affine.for %arg2 = 0 to 100 {
    // NT5-NEXT: {{.*}}affine.apply
    // NT5-NEXT: affine.for %arg3 = 0 to 1 {
    // NT5-NEXT: {{.*}}affine.apply
    // NT5-NEXT: affine.for %arg4 = 0 to 1 {
    // NT5-NEXT: {{.*}}affine.apply
    // NT5-NEXT: affine.for %arg5 = 0 to 2 {

    // NT4: affine.for %arg0 = 0 to 100 {
    // NT4-NEXT: {{.*}}affine.apply
    // NT4-NEXT: affine.for %arg1 = 0 to 100 {
    // NT4-NEXT: {{.*}}affine.apply
    // NT4-NEXT: affine.for %arg2 = 0 to 100 {
    // NT4-NEXT: {{.*}}affine.apply
    // NT4-NEXT: affine.for %arg3 = 0 to 1 {
    // NT4-NEXT: {{.*}}affine.apply
    // NT4-NEXT: affine.for %arg4 = 0 to 2 {

    // NT3: affine.for %arg0 = 0 to 100 {
    // NT3-NEXT: {{.*}}affine.apply
    // NT3-NEXT: affine.for %arg1 = 0 to 1 {
    // NT3-NEXT: {{.*}}affine.apply
    // NT3-NEXT: affine.for %arg2 = 0 to 100 {
    // NT3-NEXT: {{.*}}affine.apply
    // NT3-NEXT: affine.for %arg3 = 0 to 100 {
    // NT3-NEXT: {{.*}}affine.apply
    // NT3-NEXT: affine.for %arg4 = 0 to 2 {
    
    // NTL-NEXT: {{.*}}affine.apply
    // NTL-NEXT: {{.*}}affine.load  
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}llvm.intr.fma
    // NTL-NEXT: {{.*}}affine.store
    // NTL-NEXT: {{.*}}affine.apply
    // NTL-NEXT: {{.*}}affine.load  
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}llvm.intr.fma
    // NTL-NEXT: {{.*}}affine.store
    // NTL-NEXT: {{.*}}affine.apply
    // NTL-NEXT: {{.*}}affine.load  
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}llvm.intr.fma
    // NTL-NEXT: {{.*}}affine.store
    // NTL-NEXT: {{.*}}affine.apply
    // NTL-NEXT: {{.*}}affine.load  
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}affine.load
    // NTL-NEXT: {{.*}}llvm.intr.fma
    // NTL-NEXT: {{.*}}affine.store

    // NT2: affine.for %arg0 = 0 to 100 {
    // NT2-NEXT: {{.*}}affine.apply
    // NT2-NEXT: affine.for %arg1 = 0 to 100 {
    // NT2-NEXT: {{.*}}affine.apply
    // NT2-NEXT: affine.for %arg2 = 0 to 50 {
    // NT2-NEXT: {{.*}}affine.apply
    // NT2-NEXT: affine.for %arg3 = 0 to 2 {

    // NT1: affine.for %arg0 = 0 to 100 {
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: affine.for %arg1 = 0 to 50 {
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: affine.for %arg2 = 0 to 50 {
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: affine.for %arg3 = 0 to 2 {
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store
    // NT1-NEXT: {{.*}}affine.apply
    // NT1-NEXT: {{.*}}affine.load  
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}affine.load
    // NT1-NEXT: {{.*}}llvm.intr.fma
    // NT1-NEXT: {{.*}}affine.store

    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store
    // NTS-NEXT: {{.*}}affine.apply
    // NTS-NEXT: {{.*}}affine.load  
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}affine.load
    // NTS-NEXT: {{.*}}llvm.intr.fma
    // NTS-NEXT: {{.*}}affine.store

    
    "daphne.return"() : () -> ()
  }
}