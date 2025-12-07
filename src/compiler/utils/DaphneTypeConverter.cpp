#include "DaphneTypeConverter.h"

#include "ir/daphneir/Daphne.h"
#include "mlir/IR/BuiltinTypes.h"
#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

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

        auto dmTy = dyn_cast<daphne::MatrixType>(inputs[0].getType());
        if (!isa<RankedTensorType>(targetType) || !dmTy)
            return Value();

        if (dmTy.getRepresentation() == daphne::MatrixRepresentation::Sparse) {
            MemRefType valuesTy = MemRefType::get({ShapedType::kDynamic}, dmTy.getElementType());
            MemRefType colIdxTy = MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
            MemRefType rowPtrTy = MemRefType::get({dmTy.getNumRows() + 1}, builder.getIndexType());
            auto values = builder.create<daphne::ConvertCSRMatrixToValuesMemRef>(loc, valuesTy, inputs[0]);
            auto colIdx = builder.create<daphne::ConvertCSRMatrixToColIdxsMemRef>(loc, colIdxTy, inputs[0]);
            auto rowPtr = builder.create<daphne::ConvertCSRMatrixToRowOffsetsMemRef>(loc, rowPtrTy, inputs[0]);
            auto vt = builder.create<bufferization::ToTensorOp>(loc, memref::getTensorTypeFromMemRefType(valuesTy),
                                                                values, /*restricted=*/true);
            auto ct = builder.create<bufferization::ToTensorOp>(loc, memref::getTensorTypeFromMemRefType(colIdxTy),
                                                                colIdx, /*restricted=*/true);
            auto rt = builder.create<bufferization::ToTensorOp>(loc, memref::getTensorTypeFromMemRefType(rowPtrTy),
                                                                rowPtr, /*restricted=*/true);
            SmallVector<sparse_tensor::LevelType> CSR{sparse_tensor::LevelFormat::Dense,
                                                      sparse_tensor::LevelFormat::Compressed};
            auto enc = sparse_tensor::SparseTensorEncodingAttr::get(builder.getContext(), /*dimLevelType=*/CSR,
                                                                    /*dimOrdering=*/AffineMap(),
                                                                    /*higherOrdering=*/AffineMap(),
                                                                    /*posWidth=*/0, /*crdWidth=*/0);
            auto stTy = RankedTensorType::get({dmTy.getNumRows(), dmTy.getNumCols()}, dmTy.getElementType(), enc);
            // sparse_tensor.assemble expects level buffers in storage order:
            // positions first, then coordinates for compressed levels.
            SmallVector<Value, 3> inputs = {rt, ct};
            return builder.create<sparse_tensor::AssembleOp>(loc, stTy, inputs, vt);
        }

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
        // 1) tensor -> memref          through bufferization (or sparse accessors)
        // 2) memref -> daphne.Matrix   through conversion kernel
        if (tensorType.getEncoding()) {
            Value tensor = inputs[0];
            // Extract CSR buffers and pass raw metadata to the runtime kernel.
            Value values = builder.create<sparse_tensor::ToValuesOp>(loc, tensor);
            Value colIdxs = builder.create<sparse_tensor::ToCoordinatesOp>(loc, tensor, builder.getIndexAttr(1));
            Value rowOffsets = builder.create<sparse_tensor::ToPositionsOp>(loc, tensor, builder.getIndexAttr(1));

            auto valMeta = builder.create<memref::ExtractStridedMetadataOp>(loc, values);
            auto colMeta = builder.create<memref::ExtractStridedMetadataOp>(loc, colIdxs);
            auto rowMeta = builder.create<memref::ExtractStridedMetadataOp>(loc, rowOffsets);

            Value valBase = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, values);
            Value colBase = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, colIdxs);
            Value rowBase = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rowOffsets);

            Value rowsMem = rowMeta.getSizes()[0];
            Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
            Value rows = builder.create<arith::SubIOp>(loc, rowsMem, one);
            Value cols = builder.create<tensor::DimOp>(loc, tensor, 1);

            return builder.create<daphne::ConvertSparseBuffersToMatrix>(
                loc, targetType,
                ValueRange{valBase, valMeta.getOffset(), valMeta.getSizes()[0], valMeta.getStrides()[0], colBase,
                           colMeta.getOffset(), colMeta.getSizes()[0], colMeta.getStrides()[0], rowBase,
                           rowMeta.getOffset(), rowMeta.getSizes()[0], rowMeta.getStrides()[0], rows, cols});
        }

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

    if (matrixType.getRepresentation() == daphne::MatrixRepresentation::Sparse) {
        SmallVector<sparse_tensor::LevelType> CSR{sparse_tensor::LevelFormat::Dense,
                                                  sparse_tensor::LevelFormat::Compressed};
        auto enc = sparse_tensor::SparseTensorEncodingAttr::get(ctx, /*dimLevelType=*/CSR,
                                                                /*dimOrdering=*/AffineMap(),
                                                                /*higherOrdering=*/AffineMap(),
                                                                /*posWidth=*/0, /*crdWidth=*/0);
        return RankedTensorType::get(shape, elemTy, enc);
    }
    return RankedTensorType::get(shape, elemTy);
}
