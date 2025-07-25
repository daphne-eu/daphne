// RUN: daphne-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK

module {
  func.func @main() {
    %c0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %c5 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %c7 = "daphne.constant"() {value = 7 : si64} : () -> si64
    %c8 = "daphne.constant"() {value = 8 : si64} : () -> si64
    %c10 = "daphne.constant"() {value = 1.000000e+01 : f64} : () -> f64
    %c100 = "daphne.constant"() {value = 1.000000e+02 : f64} : () -> f64
    %cone = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %seed = "daphne.constant"() {value = 42 : si64} : () -> si64
    %true = "daphne.constant"() {value = true} : () -> i1
    %false = "daphne.constant"() {value = false} : () -> i1

    %idx7 = "daphne.cast"(%c7) : (si64) -> index
    %idx8 = "daphne.cast"(%c8) : (si64) -> index
    %idx5 = "daphne.cast"(%c5) : (si64) -> index

    %zero = "daphne.constant"() {value = 0.000000e+00 : f64} : () -> f64

    %X = "daphne.fill"(%zero, %idx7, %idx8) : (f64, index, index) -> !daphne.Matrix<7x8xf64>
    %Y = "daphne.randMatrix"(%idx5, %idx5, %c10, %c100, %cone, %seed)
      : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<5x5xf64>

    %slice = "daphne.sliceRow"(%X, %c0, %c5)
      : (!daphne.Matrix<7x8xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>

    %insertCol = "daphne.insertCol"(%slice, %Y, %c0, %c5)
      : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<5x5xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>

    %insertRow = "daphne.insertRow"(%X, %insertCol, %c0, %c5)
      : (!daphne.Matrix<7x8xf64>, !daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>

    "daphne.print"(%insertRow, %true, %false) : (!daphne.Matrix<?x?xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}

// CHECK: daphne.sliceRow
// CHECK: daphne.insertCol
// CHECK: daphne.insertRow

