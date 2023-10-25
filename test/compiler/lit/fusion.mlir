// RUN: daphne-opt -pass-pipeline="builtin.module(lower-ew, canonicalize, func.func(affine-loop-fusion))" %s | FileCheck %s""""

func.func @main() {
  %0 = "daphne.constant"() {value = 2 : index} : () -> index
  %1 = "daphne.constant"() {value = false} : () -> i1
  %2 = "daphne.constant"() {value = true} : () -> i1
  %3 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
  %4 = "daphne.constant"() {value = 2.000000e+00 : f64} : () -> f64
  %5 = "daphne.constant"() {value = 4.000000e+00 : f64} : () -> f64
  %6 = "daphne.fill"(%5, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64>
  // CHECK: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: arith.mulf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store
  %7 = "daphne.ewMul"(%6, %4) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  %8 = "daphne.ewAdd"(%7, %4) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  %9 = "daphne.ewAdd"(%7, %3) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  "daphne.print"(%7, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.print"(%8, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.print"(%9, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1) -> ()
  "daphne.return"() : () -> ()
}
