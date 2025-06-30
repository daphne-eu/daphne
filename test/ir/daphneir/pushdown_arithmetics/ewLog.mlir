// RUN: daphne-opt --canonicalize --inline %s | FileCheck %s

module {
  func.func @main() {
  }
}
