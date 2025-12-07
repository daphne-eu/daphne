#pragma once

#include "ir/daphneir/Daphne.h"
#include "mlir/Transforms/DialectConversion.h"

class DaphneTypeConverter : public mlir::TypeConverter {
  public:
    DaphneTypeConverter(mlir::MLIRContext *ctx);
    mlir::Type convertMatrixToTensor(mlir::MLIRContext *ctx, mlir::daphne::MatrixType matrixType);
};
