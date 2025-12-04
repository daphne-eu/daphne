#include "DaphneTypeConverter.h"

#include "mlir/IR/BuiltinTypes.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>

using namespace mlir;

DaphneTypeConverter::DaphneTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) -> Type { return type; });
    addConversion([ctx](IntegerType it) -> Type { return IntegerType::get(ctx, it.getWidth()); });
    addConversion([this, ctx](daphne::MatrixType type) -> Type { return convertMatrixToMemRef(ctx, type); });
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

// TODO: Maybe we should always convert daphne.Matrix to tensor types instead of memref types.
// This should at least be consistent and documented, so that certain lowerings can assume that the converted type is a
// RankedTensorType/MemRefType. Some linalg ops work only/better on tensor types. Additionally, the sparse_tensor
// dialect needs to be on the tensor type, not on memref types.
// Prob something like this:
// if (matrixType.getRepresentation() == daphne::MatrixRepresentation::Sparse) {
//     SmallVector<sparse_tensor::LevelType> CSR{sparse_tensor::LevelFormat::Dense,
//                                               sparse_tensor::LevelFormat::Compressed};
//     auto enc = sparse_tensor::SparseTensorEncodingAttr::get(ctx, [>dimLevelType=<]CSR,
//                                                             [>dimOrdering=<]AffineMap(),
//                                                             [>higherOrdering=<]AffineMap(),
//                                                             [>posWidth=*/0, /*crdWidth=<]0);
//
//     return RankedTensorType::get(shape, elemTy, enc);
// }
Type DaphneTypeConverter::convertMatrixToMemRef(MLIRContext *ctx, daphne::MatrixType matrixType) {
    auto rows = matrixType.getNumRows();
    auto cols = matrixType.getNumCols();
    SmallVector<int64_t> shape = {rows >= 0 ? rows : ShapedType::kDynamic, cols >= 0 ? cols : ShapedType::kDynamic};

    Type elemTy = matrixType.getElementType();
    if (auto it = dyn_cast<IntegerType>(elemTy))
        elemTy = convertType(elemTy);


    return MemRefType::get(shape, elemTy);
}
