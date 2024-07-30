// RUN: daphne-opt --loop-vectorization %s | FileCheck %s
module {
    // CHECK-LABEL: func.func @"e-6"
    func.func @"e-6"() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = 0 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 9 : index} : () -> index
    %5 = "daphne.constant"() {value = 187651264387696 : ui64} : () -> ui64
    %6 = "daphne.matrixConstant"(%5) : (ui64) -> !daphne.Matrix<9x1xsi64>
    %7 = "daphne.reshape"(%6, %4, %0) : (!daphne.Matrix<9x1xsi64>, index, index) -> !daphne.Matrix<9x1xsi64>
    %8 = "daphne.reshape"(%7, %1, %1) : (!daphne.Matrix<9x1xsi64>, index, index) -> !daphne.Matrix<3x3xsi64>
    %9 = "daphne.ewCos"(%8) : (!daphne.Matrix<3x3xsi64>) -> !daphne.Matrix<3x3xf64>
    %10 = "daphne.ewSin"(%8) : (!daphne.Matrix<3x3xsi64>) -> !daphne.Matrix<3x3xf64>
    // CHECK: scf.for
    // CHECK-NEXT: daphne.cast
    // CHECK-NEXT: daphne.ewMul
    // CHECK-NEXT: scf.for
    %11 = scf.for %arg0 = %2 to %1 step %0 iter_args(%arg1 = %9) -> (!daphne.Matrix<3x3xf64>) {
      %12 = "daphne.cast"(%arg0) : (index) -> si64
      %13 = "daphne.ewMul"(%12, %3) : (si64, si64) -> si64
      %14 = scf.for %arg2 = %2 to %1 step %0 iter_args(%arg3 = %arg1) -> (!daphne.Matrix<3x3xf64>) {
        %15 = "daphne.cast"(%arg2) : (index) -> si64
        %16 = "daphne.ewMul"(%15, %3) : (si64, si64) -> si64
        %17 = "daphne.ewAdd"(%13, %3) : (si64, si64) -> ui64
        %18 = "daphne.cast"(%17) : (ui64) -> si64
        %19 = "daphne.sliceRow"(%10, %13, %18) : (!daphne.Matrix<3x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %20 = "daphne.ewAdd"(%16, %3) : (si64, si64) -> ui64
        %21 = "daphne.cast"(%20) : (ui64) -> si64
        %22 = "daphne.sliceCol"(%19, %16, %21) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %23 = "daphne.cast"(%20) : (ui64) -> si64
        %24 = "daphne.sliceRow"(%arg3, %16, %23) : (!daphne.Matrix<3x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %25 = "daphne.cast"(%17) : (ui64) -> si64
        %26 = "daphne.insertCol"(%24, %22, %13, %25) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %27 = "daphne.cast"(%20) : (ui64) -> si64
        %28 = "daphne.insertRow"(%arg3, %26, %16, %27) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<3x3xf64>
        scf.yield %28 : !daphne.Matrix<3x3xf64>
      }
      scf.yield %14 : !daphne.Matrix<3x3xf64>
    }
    "daphne.return"() : () -> ()
  }
  // CHECK-LABEL: func.func @"f-5"
  func.func @"f-5"() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = 0 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 2 : index} : () -> index
    %5 = "daphne.constant"() {value = 6 : index} : () -> index
    %6 = "daphne.constant"() {value = 187651263218944 : ui64} : () -> ui64
    %7 = "daphne.matrixConstant"(%6) : (ui64) -> !daphne.Matrix<6x1xsi64>
    %8 = "daphne.reshape"(%7, %5, %0) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<6x1xsi64>
    %9 = "daphne.reshape"(%8, %4, %1) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.ewCos"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    %11 = "daphne.ewSin"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    // CHECK-NO: scf.for
    %12 = scf.for %arg0 = %2 to %1 step %0 iter_args(%arg1 = %10) -> (!daphne.Matrix<2x3xf64>) {
      %13 = "daphne.cast"(%arg0) : (index) -> si64
      %14 = "daphne.ewMul"(%13, %3) : (si64, si64) -> si64
      %15 = scf.for %arg2 = %2 to %4 step %0 iter_args(%arg3 = %arg1) -> (!daphne.Matrix<2x3xf64>) {
        %16 = "daphne.cast"(%arg2) : (index) -> si64
        %17 = "daphne.ewMul"(%16, %3) : (si64, si64) -> si64
        %18 = "daphne.ewAdd"(%17, %3) : (si64, si64) -> ui64
        %19 = "daphne.cast"(%18) : (ui64) -> si64
        %20 = "daphne.sliceRow"(%11, %17, %19) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %21 = "daphne.ewAdd"(%14, %3) : (si64, si64) -> ui64
        %22 = "daphne.cast"(%21) : (ui64) -> si64
        %23 = "daphne.sliceCol"(%20, %14, %22) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.cast"(%18) : (ui64) -> si64
        %25 = "daphne.sliceRow"(%arg3, %17, %24) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %26 = "daphne.cast"(%21) : (ui64) -> si64
        %27 = "daphne.insertCol"(%25, %23, %14, %26) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %28 = "daphne.cast"(%18) : (ui64) -> si64
        %29 = "daphne.insertRow"(%arg3, %27, %17, %28) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
        scf.yield %29 : !daphne.Matrix<2x3xf64>
      }
      scf.yield %15 : !daphne.Matrix<2x3xf64>
    }
    "daphne.return"() : () -> ()
  }
  // CHECK-LABEL: func.func @"d-4"
  func.func @"d-4"() {
    %0 = "daphne.constant"() {value = 3 : index} : () -> index
    %1 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %2 = "daphne.constant"() {value = 2 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : index} : () -> index
    %4 = "daphne.constant"() {value = 11 : index} : () -> index
    %5 = "daphne.constant"() {value = 0 : index} : () -> index
    %6 = "daphne.constant"() {value = 6 : index} : () -> index
    %7 = "daphne.constant"() {value = 187651262036512 : ui64} : () -> ui64
    %8 = "daphne.matrixConstant"(%7) : (ui64) -> !daphne.Matrix<6x1xsi64>
    %9 = "daphne.reshape"(%8, %6, %3) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<6x1xsi64>
    %10 = "daphne.reshape"(%9, %2, %0) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<2x3xsi64>
    %11 = "daphne.ewCos"(%10) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    // CHECK: scf.for
    // CHECK-NEXT: daphne.cast
    // CHECK-NEXT: daphne.ewAdd
    %12 = scf.for %arg0 = %5 to %4 step %3 iter_args(%arg1 = %11) -> (!daphne.Matrix<2x3xf64>) {
      %13 = "daphne.cast"(%arg0) : (index) -> si64
      %14 = scf.for %arg2 = %5 to %0 step %3 iter_args(%arg3 = %arg1) -> (!daphne.Matrix<2x3xf64>) {
        %15 = "daphne.cast"(%arg2) : (index) -> si64
        %16 = "daphne.ewMul"(%15, %1) : (si64, si64) -> si64
        %17 = scf.for %arg4 = %5 to %2 step %3 iter_args(%arg5 = %arg3) -> (!daphne.Matrix<2x3xf64>) {
          %18 = "daphne.cast"(%arg4) : (index) -> si64
          %19 = "daphne.ewMul"(%18, %1) : (si64, si64) -> si64
          %20 = "daphne.ewAdd"(%19, %1) : (si64, si64) -> ui64
          %21 = "daphne.cast"(%20) : (ui64) -> si64
          %22 = "daphne.sliceRow"(%arg5, %19, %21) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
          %23 = "daphne.ewAdd"(%16, %1) : (si64, si64) -> ui64
          %24 = "daphne.cast"(%23) : (ui64) -> si64
          %25 = "daphne.sliceCol"(%22, %16, %24) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
          %26 = "daphne.ewAdd"(%25, %1) : (!daphne.Matrix<?x?xf64>, si64) -> !daphne.Matrix<?x?xf64>
          %27 = "daphne.cast"(%20) : (ui64) -> si64
          %28 = "daphne.sliceRow"(%arg5, %19, %27) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
          %29 = "daphne.cast"(%23) : (ui64) -> si64
          %30 = "daphne.insertCol"(%28, %26, %16, %29) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
          %31 = "daphne.cast"(%20) : (ui64) -> si64
          %32 = "daphne.insertRow"(%arg5, %30, %19, %31) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
          scf.yield %32 : !daphne.Matrix<2x3xf64>
        }
        scf.yield %17 : !daphne.Matrix<2x3xf64>
      }
      scf.yield %14 : !daphne.Matrix<2x3xf64>
    }
    "daphne.return"() : () -> ()
  }
  // CHECK-LABEL: func.func @"c-3"
  func.func @"c-3"() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 0 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 3 : index} : () -> index
    %5 = "daphne.constant"() {value = 6 : index} : () -> index
    %6 = "daphne.constant"() {value = false} : () -> i1
    %7 = "daphne.constant"() {value = true} : () -> i1
    %8 = "daphne.constant"() {value = 187651262016688 : ui64} : () -> ui64
    %9 = "daphne.matrixConstant"(%8) : (ui64) -> !daphne.Matrix<6x1xsi64>
    %10 = "daphne.reshape"(%9, %5, %0) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<6x1xsi64>
    %11 = "daphne.reshape"(%10, %1, %4) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<2x3xsi64>
    %12 = "daphne.ewCos"(%11) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    %13 = "daphne.ewSin"(%11) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    // CHECK: scf.for
    // CHECK-NEXT: daphne.cast
    // CHECK-NEXT: daphne.ewMul
    // CHECK-NEXT: scf.for
    %14 = scf.for %arg0 = %2 to %1 step %0 iter_args(%arg1 = %12) -> (!daphne.Matrix<2x3xf64>) {
      %15 = "daphne.cast"(%arg0) : (index) -> si64
      %16 = "daphne.ewMul"(%15, %3) : (si64, si64) -> si64
      %17 = scf.for %arg2 = %2 to %4 step %0 iter_args(%arg3 = %arg1) -> (!daphne.Matrix<2x3xf64>) {
        %18 = "daphne.cast"(%arg2) : (index) -> si64
        %19 = "daphne.ewMul"(%18, %3) : (si64, si64) -> si64
        "daphne.print"(%16, %7, %6) : (si64, i1, i1) -> ()
        %20 = "daphne.ewAdd"(%16, %3) : (si64, si64) -> ui64
        %21 = "daphne.cast"(%20) : (ui64) -> si64
        %22 = "daphne.sliceRow"(%13, %16, %21) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %23 = "daphne.ewAdd"(%19, %3) : (si64, si64) -> ui64
        %24 = "daphne.cast"(%23) : (ui64) -> si64
        %25 = "daphne.sliceCol"(%22, %19, %24) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %26 = "daphne.cast"(%20) : (ui64) -> si64
        %27 = "daphne.sliceRow"(%arg3, %16, %26) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %28 = "daphne.cast"(%23) : (ui64) -> si64
        %29 = "daphne.insertCol"(%27, %25, %19, %28) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %30 = "daphne.cast"(%20) : (ui64) -> si64
        %31 = "daphne.insertRow"(%arg3, %29, %16, %30) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
        scf.yield %31 : !daphne.Matrix<2x3xf64>
      }
      scf.yield %17 : !daphne.Matrix<2x3xf64>
    }
    "daphne.return"() : () -> ()
  }
  
  // CHECK-LABEL: func.func @"b-2"
  func.func @"b-2"() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 0 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 3 : index} : () -> index
    %5 = "daphne.constant"() {value = 6 : index} : () -> index
    %6 = "daphne.constant"() {value = 187651262003680 : ui64} : () -> ui64
    %7 = "daphne.matrixConstant"(%6) : (ui64) -> !daphne.Matrix<6x1xsi64>
    %8 = "daphne.reshape"(%7, %5, %0) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<6x1xsi64>
    %9 = "daphne.reshape"(%8, %1, %4) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.ewCos"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    %11 = "daphne.ewSin"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: scf.for
    // CHECK: daphne.ewSin
    // CHECK-NEXT: daphne.return
    %12 = scf.for %arg0 = %2 to %1 step %0 iter_args(%arg1 = %10) -> (!daphne.Matrix<2x3xf64>) {
      %13 = "daphne.cast"(%arg0) : (index) -> si64
      %14 = "daphne.ewMul"(%13, %3) : (si64, si64) -> si64
      %15 = scf.for %arg2 = %2 to %4 step %0 iter_args(%arg3 = %arg1) -> (!daphne.Matrix<2x3xf64>) {
        %16 = "daphne.cast"(%arg2) : (index) -> si64
        %17 = "daphne.ewMul"(%16, %3) : (si64, si64) -> si64
        %18 = "daphne.ewAdd"(%14, %3) : (si64, si64) -> ui64
        %19 = "daphne.cast"(%18) : (ui64) -> si64
        %20 = "daphne.sliceRow"(%11, %14, %19) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %21 = "daphne.ewAdd"(%17, %3) : (si64, si64) -> ui64
        %22 = "daphne.cast"(%21) : (ui64) -> si64
        %23 = "daphne.sliceCol"(%20, %17, %22) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.cast"(%18) : (ui64) -> si64
        %25 = "daphne.sliceRow"(%arg3, %14, %24) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %26 = "daphne.cast"(%21) : (ui64) -> si64
        %27 = "daphne.insertCol"(%25, %23, %17, %26) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %28 = "daphne.cast"(%18) : (ui64) -> si64
        %29 = "daphne.insertRow"(%arg3, %27, %14, %28) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
        scf.yield %29 : !daphne.Matrix<2x3xf64>
      }
      scf.yield %15 : !daphne.Matrix<2x3xf64>
    }
    "daphne.return"() : () -> ()
  }

  // CHECK-LABEL: func.func @"a-1"
  func.func @"a-1"() {
    %0 = "daphne.constant"() {value = 1 : index} : () -> index
    %1 = "daphne.constant"() {value = 2 : index} : () -> index
    %2 = "daphne.constant"() {value = 0 : index} : () -> index
    %3 = "daphne.constant"() {value = 1 : si64} : () -> si64
    %4 = "daphne.constant"() {value = 3 : index} : () -> index
    %5 = "daphne.constant"() {value = 6 : index} : () -> index
    %6 = "daphne.constant"() {value = 187651261662912 : ui64} : () -> ui64
    %7 = "daphne.matrixConstant"(%6) : (ui64) -> !daphne.Matrix<6x1xsi64>
    %8 = "daphne.reshape"(%7, %5, %0) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<6x1xsi64>
    %9 = "daphne.reshape"(%8, %1, %4) : (!daphne.Matrix<6x1xsi64>, index, index) -> !daphne.Matrix<2x3xsi64>
    %10 = "daphne.ewCos"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    %11 = "daphne.ewSin"(%9) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xf64>
    // CHECK-NOT: scf.for
    // CHECK: daphne.ewAbs
    // CHECK-NEXT: daphne.ewSqrt
    // CHECK-NEXT: daphne.ewAdd
    %12:2 = scf.for %arg0 = %2 to %1 step %0 iter_args(%arg1 = %11, %arg2 = %10) -> (!daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>) {
      %13 = "daphne.cast"(%arg0) : (index) -> si64
      %14 = "daphne.ewMul"(%13, %3) : (si64, si64) -> si64
      %15:2 = scf.for %arg3 = %2 to %4 step %0 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (!daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>) {
        %16 = "daphne.cast"(%arg3) : (index) -> si64
        %17 = "daphne.ewMul"(%16, %3) : (si64, si64) -> si64
        %18 = "daphne.ewAdd"(%14, %3) : (si64, si64) -> ui64
        %19 = "daphne.cast"(%18) : (ui64) -> si64
        %20 = "daphne.sliceRow"(%arg5, %14, %19) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %21 = "daphne.ewAdd"(%17, %3) : (si64, si64) -> ui64
        %22 = "daphne.cast"(%21) : (ui64) -> si64
        %23 = "daphne.sliceCol"(%20, %17, %22) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %24 = "daphne.cast"(%18) : (ui64) -> si64
        %25 = "daphne.sliceRow"(%arg5, %14, %24) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %26 = "daphne.cast"(%21) : (ui64) -> si64
        %27 = "daphne.sliceCol"(%25, %17, %26) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %28 = "daphne.ewAbs"(%27) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %29 = "daphne.ewSqrt"(%28) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %30 = "daphne.ewAdd"(%23, %29) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %31 = "daphne.cast"(%18) : (ui64) -> si64
        %32 = "daphne.sliceRow"(%arg4, %14, %31) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %33 = "daphne.cast"(%21) : (ui64) -> si64
        %34 = "daphne.sliceCol"(%32, %17, %33) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %35 = "daphne.ewCos"(%34) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %36 = "daphne.ewAdd"(%30, %35) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
        %37 = "daphne.cast"(%18) : (ui64) -> si64
        %38 = "daphne.sliceRow"(%arg4, %14, %37) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %39 = "daphne.cast"(%21) : (ui64) -> si64
        %40 = "daphne.insertCol"(%38, %36, %17, %39) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %41 = "daphne.cast"(%18) : (ui64) -> si64
        %42 = "daphne.insertRow"(%arg4, %40, %14, %41) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
        %43 = "daphne.cast"(%18) : (ui64) -> si64
        %44 = "daphne.sliceRow"(%42, %14, %43) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %45 = "daphne.cast"(%21) : (ui64) -> si64
        %46 = "daphne.sliceCol"(%44, %17, %45) : (!daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
        %47 = "daphne.cast"(%18) : (ui64) -> si64
        %48 = "daphne.sliceRow"(%arg5, %14, %47) : (!daphne.Matrix<2x3xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %49 = "daphne.cast"(%21) : (ui64) -> si64
        %50 = "daphne.insertCol"(%48, %46, %17, %49) : (!daphne.Matrix<?x3xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x3xf64>
        %51 = "daphne.cast"(%18) : (ui64) -> si64
        %52 = "daphne.insertRow"(%arg5, %50, %14, %51) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<?x3xf64>, si64, si64) -> !daphne.Matrix<2x3xf64>
        scf.yield %42, %52 : !daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>
      }
      scf.yield %15#0, %15#1 : !daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>
    }
    "daphne.return"() : () -> ()
  }
}