// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK -dump-input=always

module {
    func.func @main() {
        // X @ transpose(Y)
        %rX    = "daphne.constant"() {value = 4 : index} : () -> index
        %cX    = "daphne.constant"() {value = 6 : index} : () -> index
        %cY    = "daphne.constant"() {value = 6 : index} : () -> index
        %low   = "daphne.constant"() {value = 0.0 : f64} : () -> f64
        %high  = "daphne.constant"() {value = 1.0 : f64} : () -> f64
        %fill  = "daphne.constant"() {value = 1.0 : f64} : () -> f64
        %seed  = "daphne.constant"() {value = 42 : si64} : () -> si64

        %X     = "daphne.randMatrix"(%rX, %cX, %low, %high, %fill, %seed)
                 : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x6xf64>
        %Y     = "daphne.randMatrix"(%cX, %cY, %low, %high, %fill, %seed)
                 : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<6x5xf64>

        %t     = "daphne.constant"() {value = true} : () -> i1
        %f     = "daphne.constant"() {value = false} : () -> i1

        %matmul = "daphne.matMul"(%X, %Y, %f, %t)
                  : (!daphne.Matrix<4x6xf64>, !daphne.Matrix<6x5xf64>, i1, i1) -> !daphne.Matrix<4x5xf64>

        %i3    = "daphne.constant"() {value = 3 : si64} : () -> si64
        %i1    = "daphne.constant"() {value = 4 : si64} : () -> si64
        %i3p   = "daphne.constant"() {value = 4 : si64} : () -> si64
        %i1p   = "daphne.constant"() {value = 5 : si64} : () -> si64

        %slice_row = "daphne.sliceRow"(%matmul, %i3, %i3p)
                     : (!daphne.Matrix<4x5xf64>, si64, si64) -> !daphne.Matrix<1x5xf64>
        %slice_col = "daphne.sliceCol"(%slice_row, %i1, %i1p)
                     : (!daphne.Matrix<1x5xf64>, si64, si64) -> !daphne.Matrix<1x1xf64>

        %bool_1 = "daphne.constant"() {value = true} : () -> i1
        %bool_2 = "daphne.constant"() {value = false} : () -> i1

        "daphne.print"(%slice_col, %bool_1, %bool_2) : (!daphne.Matrix<1x1xf64>, i1, i1) -> ()
        "daphne.return"() : () -> ()
    }
}

// CHECK: %[[X:.*]] = "daphne.randMatrix"({{.*}}) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<4x6xf64>
// CHECK: %[[Y:.*]] = "daphne.randMatrix"({{.*}}) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<6x5xf64>
// CHECK: %[[ROW:.*]] = "daphne.sliceRow"(%[[X]], {{.*}}, {{.*}}) : (!daphne.Matrix<4x6xf64>, si64, si64) -> !daphne.Unknown
// CHECK: %[[COL:.*]] = "daphne.sliceRow"(%[[Y]], {{.*}}, {{.*}}) : (!daphne.Matrix<6x5xf64>, si64, si64) -> !daphne.Unknown
// CHECK: "daphne.matMul"(%[[ROW]], %[[COL]], {{.*}}, {{.*}}) : (!daphne.Unknown, !daphne.Unknown, i1, i1) -> !daphne.Matrix<1x1xf64>
// CHECK-NOT: "daphne.sliceCol"(%{{.*matMul.*}})
// CHECK-NOT: "daphne.sliceRow"(%{{.*matMul.*}})
