// RUN: daphne-opt --canonicalize %s | FileCheck %s

func.func @string_string() {
  %0 = "daphne.constant"() {value = "debug"} : () -> !daphne.String
  %1 = "daphne.constant"() {value = "debug"} : () -> !daphne.String
  // CHECK-NOT: daphne.ewEq
  %2 = "daphne.ewEq"(%0, %1) : (!daphne.String, !daphne.String) -> !daphne.String
  %3 = "daphne.cast"(%2) : (!daphne.String) -> i1
  "daphne.print"(%0, %3, %3) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @string_int() {
    // CHECK-NOT: daphne.eqEq
    // CHECK: daphne.cast
    // CHECK: daphne.stringEq
    %0 = "daphne.constant"() {value = "debug"} : () -> !daphne.String
    %1 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %2 = "daphne.ewEq"(%0, %1) : (!daphne.String, si64) -> !daphne.String
    %3 = "daphne.cast"(%2) : (!daphne.String) -> i1
  "daphne.print"(%0, %3, %3) : (!daphne.String, i1, i1) -> ()
  "daphne.return"() : () -> ()
}

func.func @int_int_do_not_canonicalize() {
  %0 = "daphne.constant"() {value = 2 : si64} : () -> si64
  %1 = "daphne.constant"() {value = 5 : si64} : () -> si64
  %2 = "daphne.ewEq"(%0, %1) : (si64, si64) -> si64
  // CHECK-NOT: daphne.stringEq
  %3 = "daphne.cast"(%2) : (si64) -> i1
  scf.if %3 {
    %4 = "daphne.constant"() {value = "debug"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%4, %5, %6) : (!daphne.String, i1, i1) -> ()
  } else {
    %4 = "daphne.constant"() {value = "release"} : () -> !daphne.String
    %5 = "daphne.constant"() {value = true} : () -> i1
    %6 = "daphne.constant"() {value = false} : () -> i1
    "daphne.print"(%4, %5, %6) : (!daphne.String, i1, i1) -> ()
  }
  "daphne.return"() : () -> ()
}
