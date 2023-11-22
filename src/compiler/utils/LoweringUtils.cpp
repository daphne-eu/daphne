/*
 * Copyright 2023 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LoweringUtils.h"

#include <ir/daphneir/Passes.h>

#include "ir/daphneir/Daphne.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/Passes.h"

/// Insert an allocation for the given MemRefType.
mlir::Value insertMemRefAlloc(mlir::MemRefType type, mlir::Location loc,
                              mlir::PatternRewriter &rewriter) {
    auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    return alloc;
}

void insertMemRefDealloc(mlir::Value memref, mlir::Location loc,
                         mlir::PatternRewriter &rewriter) {
    auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, memref);
    dealloc->moveBefore(&memref.getParentBlock()->back());
}

// TODO(phil) try to provide function templates to remove duplication
void affineFillMemRefInt(int value, mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                         mlir::MLIRContext *ctx, mlir::Value memRef,
                         mlir::Type elemType) {
    constexpr int ROW = 0;
    constexpr int COL = 1;
    mlir::Value fillValue = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(value));

    llvm::SmallVector<mlir::Value, 4> loopIvs;

    auto outerLoop = rewriter.create<mlir::AffineForOp>(loc, 0, shape[ROW], 1);
    for (mlir::Operation &nested : *outerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(outerLoop.getInductionVar());

    // outer loop body
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = rewriter.create<mlir::AffineForOp>(loc, 0, shape[COL], 1);
    for (mlir::Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(innerLoop.getInductionVar());
    rewriter.create<mlir::AffineYieldOp>(loc);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    rewriter.create<mlir::AffineStoreOp>(loc, fillValue, memRef, loopIvs);

    rewriter.create<mlir::AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(outerLoop);
}

void affineFillMemRef(double value, mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                      mlir::MLIRContext *ctx, mlir::Value memRef,
                      mlir::Type elemType) {
    constexpr int ROW = 0;
    constexpr int COL = 1;
    mlir::Value fillValue = rewriter.create<mlir::arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, value));

    llvm::SmallVector<mlir::Value, 4> loopIvs;

    auto outerLoop = rewriter.create<mlir::AffineForOp>(loc, 0, shape[ROW], 1);
    for (mlir::Operation &nested : *outerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(outerLoop.getInductionVar());

    // outer loop body
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = rewriter.create<mlir::AffineForOp>(loc, 0, shape[COL], 1);
    for (mlir::Operation &nested : *innerLoop.getBody()) {
        rewriter.eraseOp(&nested);
    }
    loopIvs.push_back(innerLoop.getInductionVar());
    rewriter.create<mlir::AffineYieldOp>(loc);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    rewriter.create<mlir::AffineStoreOp>(loc, fillValue, memRef, loopIvs);

    rewriter.create<mlir::AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(outerLoop);
}

mlir::Value convertMemRefToDenseMatrix(
    mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
    mlir::Value memRef, mlir::Type type) {
    auto extractStridedMetadataOp =
        rewriter.create<mlir::memref::ExtractStridedMetadataOp>(loc, memRef);
    // aligned ptr (memref.data)
    mlir::Value alignedPtr =
        rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                      memRef);
    // offset
    mlir::Value offset = extractStridedMetadataOp.getOffset();
    // strides
    mlir::ResultRange strides = extractStridedMetadataOp.getStrides();
    // sizes
    mlir::ResultRange sizes = extractStridedMetadataOp.getSizes();

    return rewriter.create<mlir::daphne::ConvertMemRefToDenseMatrix>(
        loc, type, alignedPtr, offset, sizes[0], sizes[1], strides[0],
        strides[1]);
}

mlir::Type convertFloat(mlir::FloatType floatType) {
    return mlir::IntegerType::get(floatType.getContext(),
                                  floatType.getIntOrFloatBitWidth());
}

mlir::Type convertInteger(mlir::IntegerType intType) {
    return mlir::IntegerType::get(intType.getContext(),
                                  intType.getIntOrFloatBitWidth());
}

llvm::Optional<mlir::Value> materializeCastFromIllegal(mlir::OpBuilder &builder,
                                                       mlir::Type type,
                                                       mlir::ValueRange inputs,
                                                       mlir::Location loc) {
    mlir::Type fromType = getElementTypeOrSelf(inputs[0].getType());
    mlir::Type toType = getElementTypeOrSelf(type);

    if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
        !toType.isSignlessInteger())
        return std::nullopt;
    // Use unrealized conversion casts to do signful->signless conversions.
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
}

llvm::Optional<mlir::Value> materializeCastToIllegal(mlir::OpBuilder &builder,
                                                     mlir::Type type,
                                                     mlir::ValueRange inputs,
                                                     mlir::Location loc) {
    mlir::Type fromType = getElementTypeOrSelf(inputs[0].getType());
    mlir::Type toType = getElementTypeOrSelf(type);

    if (!fromType.isSignlessInteger() ||
        (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
        return std::nullopt;
    // Use unrealized conversion casts to do signless->signful conversions.
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
}

mlir::Operation *findLastUseOfSSAValue(mlir::Value &v) {
    mlir::Operation *lastUseOp = nullptr;

    for (mlir::OpOperand &use : v.getUses()) {
        mlir::Operation *thisUseOp = use.getOwner();
        // Find parent op in the block where v is defined.
        while (thisUseOp->getBlock() != v.getParentBlock())
            thisUseOp = thisUseOp->getParentOp();
        // Determine if this is a later use.
        if (!lastUseOp || lastUseOp->isBeforeInBlock(thisUseOp))
            lastUseOp = thisUseOp;
    }

    return lastUseOp;
}
