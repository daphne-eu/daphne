// RUN: daphne-opt --lower-mm %s | FileCheck %s

// double
module {
  func.func @double() {
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
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-SAME: (f64, f64, f64) -> f64
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
    "daphne.return"() : () -> ()
  }
}

// single
module {
  func.func @single() {
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
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}affine.load
    // CHECK-NEXT: {{.*}}llvm.intr.fma
    // CHECK-SAME: (f32, f32, f32) -> f32
    // CHECK-NEXT: {{.*}}affine.store
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf32>, !daphne.Matrix<10x10xf32>, i1, i1) -> !daphne.Matrix<10x10xf32>
    "daphne.return"() : () -> ()
  }
}

// integer
module {
  func.func @signedInteger() {
    
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : si32} : () -> si32
    %3 = "daphne.constant"() {value = 5 : si32} : () -> si32
    %4 = "daphne.fill"(%3, %0, %0) : (si32, index, index) -> !daphne.Matrix<10x10xsi32>
    %5 = "daphne.fill"(%2, %0, %0) : (si32, index, index) -> !daphne.Matrix<10x10xsi32>
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
    // CHECK: {{.*}}affine.store
    // CHECK-SAME: memref<10x10xsi32>
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xsi32>, !daphne.Matrix<10x10xsi32>, i1, i1) -> !daphne.Matrix<10x10xsi32>
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @unsignedInteger() {
    
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : ui32} : () -> ui32
    %3 = "daphne.constant"() {value = 5 : ui32} : () -> ui32
    %4 = "daphne.fill"(%3, %0, %0) : (ui32, index, index) -> !daphne.Matrix<10x10xui32>
    %5 = "daphne.fill"(%2, %0, %0) : (ui32, index, index) -> !daphne.Matrix<10x10xui32>
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
    // CHECK: {{.*}}affine.store
    // CHECK-SAME: memref<10x10xui32>
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xui32>, !daphne.Matrix<10x10xui32>, i1, i1) -> !daphne.Matrix<10x10xui32>
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @nonSquare() {
    
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = 3 : ui32} : () -> ui32
    %3 = "daphne.constant"() {value = 5 : ui32} : () -> ui32
    %4 = "daphne.fill"(%3, %0, %0) : (ui32, index, index) -> !daphne.Matrix<10x5xui32>
    %5 = "daphne.fill"(%2, %0, %0) : (ui32, index, index) -> !daphne.Matrix<5x10xui32>
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
    // CHECK: {{.*}}affine.store
    // CHECK-SAME: memref<10x10xui32>
    %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x5xui32>, !daphne.Matrix<5x10xui32>, i1, i1) -> !daphne.Matrix<10x10xui32>
    "daphne.return"() : () -> ()
  }
}