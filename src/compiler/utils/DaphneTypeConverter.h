#pragma once

#include "ir/daphneir/Daphne.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class DaphneTypeConverter : public TypeConverter {
  public:
    DaphneTypeConverter(MLIRContext *ctx);
    Type convertMatrixToMemRef(daphne::MatrixType matrixType);
};
