#include "DaphneTypeConverter.h"

#include "mlir/IR/BuiltinTypes.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace mlir;

DaphneTypeConverter::DaphneTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) -> Type { return type; });
    addConversion([this](daphne::MatrixType type) -> Type { return convertMatrixToMemRef(type); });
    addTargetMaterialization([](OpBuilder &builder, Type targetType, ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
            return Value();
        return builder.create<daphne::ConvertDenseMatrixToMemRef>(loc, targetType, inputs[0]);
    });

    addSourceMaterialization([](OpBuilder &builder, Type targetType, ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
            return Value();
        auto matrixType = dyn_cast<daphne::MatrixType>(targetType);
        auto memrefType = dyn_cast<MemRefType>(inputs[0].getType());
        if (!matrixType || !memrefType)
            return Value();

        Value memref = inputs[0];
        auto meta = builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
        Value basePtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
        Value offset = meta.getOffset();
        Value size0 = meta.getSizes()[0];
        Value size1 = meta.getSizes()[1];
        Value stride0 = meta.getStrides()[0];
        Value stride1 = meta.getStrides()[1];

        return builder.create<daphne::ConvertMemRefToDenseMatrix>(
            loc, targetType, ValueRange{basePtr, offset, size0, size1, stride0, stride1});
    });
}

Type DaphneTypeConverter::convertMatrixToMemRef(daphne::MatrixType matrixType) {
    auto rows = matrixType.getNumRows();
    auto cols = matrixType.getNumCols();

    // return MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, matrixType.getElementType());
    return MemRefType::get({rows, cols}, matrixType.getElementType());
}

