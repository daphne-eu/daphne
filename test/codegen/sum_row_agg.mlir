// RUN: daphne-opt --lower-agg-row %s | FileCheck %s

module {
  func.func @double() {
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
    // CHECK-NOT: sumRow
    
    // COM: Checks conversions (and dimension) at beginning and end of loop and basic loop body
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<10x10xf64>

    // CHECK: memref.alloc
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: arith.addf
    // CHECK: affine.store
    
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<10x1xf64>
    %5 = "daphne.sumRow"(%4) : (!daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x1xf64>
    "daphne.print"(%5, %2, %1) : (!daphne.Matrix<10x1xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @singlePrecisionFloat() {
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %4 = "daphne.fill"(%3, %0, %0) : (f32, index, index) -> !daphne.Matrix<10x10xf32>
    // CHECK-NOT: sumRow
    
    // COM: Checks conversions (and dimension) at beginning and end of loop and basic loop body
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<10x10xf32>

    // CHECK: memref.alloc
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: arith.addf
    // CHECK: affine.store
    
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<10x1xf32>
    %5 = "daphne.sumRow"(%4) : (!daphne.Matrix<10x10xf32>) -> !daphne.Matrix<10x1xf32>
    "daphne.print"(%5, %2, %1) : (!daphne.Matrix<10x1xf32>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @signedIntegers() {
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.fill"(%3, %0, %0) : (si64, index, index) -> !daphne.Matrix<10x10xsi64>
    // CHECK-NOT: sumRow
    
    // COM: Checks conversions (and dimension) at beginning and end of loop and basic loop body
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<10x10xsi64>

    // CHECK: memref.alloc
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: builtin.unrealized_conversion_cast
    // CHECK-NEXT: arith.addi
    // CHECK: affine.store
    
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<10x1xsi64>
    %5 = "daphne.sumRow"(%4) : (!daphne.Matrix<10x10xsi64>) -> !daphne.Matrix<10x1xsi64>
    "daphne.print"(%5, %2, %1) : (!daphne.Matrix<10x1xsi64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

module {
  func.func @unsignedIntegers() {
    %0 = "daphne.constant"() {value = 10 : index} : () -> index
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.constant"() {value = true} : () -> i1
    %3 = "daphne.constant"() {value = 1 : ui64} : () -> ui64
    %4 = "daphne.fill"(%3, %0, %0) : (ui64, index, index) -> !daphne.Matrix<10x10xui64>
    // CHECK-NOT: sumRow
    
    // COM: Checks conversions (and dimension) at beginning and end of loop and basic loop body
    // CHECK: "daphne.convertDenseMatrixToMemRef"{{\(.*\) : \(.*\)}} -> memref<10x10xui64>

    // CHECK: memref.alloc
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: affine.for
    // CHECK-NEXT: affine.load
    // CHECK-NEXT: builtin.unrealized_conversion_cast
    // CHECK-NEXT: arith.addi
    // CHECK: affine.store
    
    // CHECK: "daphne.convertMemRefToDenseMatrix"{{\(.*\) : \(.*\)}} -> !daphne.Matrix<10x1xui64>
    %5 = "daphne.sumRow"(%4) : (!daphne.Matrix<10x10xui64>) -> !daphne.Matrix<10x1xui64>
    "daphne.print"(%5, %2, %1) : (!daphne.Matrix<10x1xui64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}