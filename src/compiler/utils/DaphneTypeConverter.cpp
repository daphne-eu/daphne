#include "DaphneTypeConverter.h"

#include "mlir/IR/BuiltinTypes.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>

using namespace mlir;

DaphneTypeConverter::DaphneTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) -> Type { return type; });
    addConversion([ctx](IntegerType it) -> Type { return IntegerType::get(ctx, it.getWidth()); });
    addConversion([this, ctx](daphne::MatrixType type) -> Type { return convertMatrixToTensor(ctx, type); });
    addTargetMaterialization([](OpBuilder &builder, Type targetType, ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
            return Value();

        if (auto dstInt = dyn_cast<IntegerType>(targetType)) {
            if (isa<IntegerType>(inputs[0].getType()))
                return builder.create<UnrealizedConversionCastOp>(loc, dstInt, inputs[0]).getResult(0);
        }

        if (!isa<RankedTensorType>(targetType) || !isa<daphne::MatrixType>(inputs[0].getType()))
            return Value();
        RankedTensorType tensorType = dyn_cast<RankedTensorType>(targetType);
        auto memrefTy = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        Value cdm2m = builder.create<daphne::ConvertDenseMatrixToMemRef>(loc, memrefTy, inputs[0]);
        return builder.create<bufferization::ToTensorOp>(loc, targetType, cdm2m, /*restricted=*/true);
    });

    addSourceMaterialization([](OpBuilder &builder, Type targetType, ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
            return Value();

        if (auto dstInt = dyn_cast<IntegerType>(targetType)) {
            if (isa<IntegerType>(inputs[0].getType()))
                return builder.create<UnrealizedConversionCastOp>(loc, targetType, inputs[0]).getResult(0);
        }

        auto matrixType = dyn_cast<daphne::MatrixType>(targetType);
        RankedTensorType tensorType = dyn_cast<RankedTensorType>(inputs[0].getType());
        if (!matrixType || !tensorType)
            return Value();

        // Materialization for tensor<...> to daphne.Matrix<...>
        // 1) tensor -> memref          through bufferization
        // 2) memref -> daphne.Matrix   through conversion kernel
        auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        Value memref = builder.create<bufferization::ToBufferOp>(loc, memRefType, inputs[0]);
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

Type DaphneTypeConverter::convertMatrixToTensor(MLIRContext *ctx, daphne::MatrixType matrixType) {
    auto rows = matrixType.getNumRows();
    auto cols = matrixType.getNumCols();
    SmallVector<int64_t> shape = {rows >= 0 ? rows : ShapedType::kDynamic, cols >= 0 ? cols : ShapedType::kDynamic};

    Type elemTy = matrixType.getElementType();
    if (auto it = dyn_cast<IntegerType>(elemTy))
        elemTy = convertType(elemTy);

    return RankedTensorType::get(shape, elemTy);
}
