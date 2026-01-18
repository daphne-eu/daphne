#pragma once

#include "ir/daphneir/Daphne.h"
#include "mlir/Transforms/DialectConversion.h"

class DaphneTypeConverter : public mlir::TypeConverter {
    mlir::Type convertMatrixToTensor(mlir::MLIRContext *ctx, mlir::daphne::MatrixType matrixType);

  public:
    DaphneTypeConverter(mlir::MLIRContext *ctx);
};
