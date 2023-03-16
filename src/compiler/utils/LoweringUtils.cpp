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
#include "ir/daphneir/Daphne.h"

/// Insert an allocation and deallocation for the given MemRefType.
mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                  mlir::PatternRewriter &rewriter) {
    auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block.
    auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}

// TODO(phil): Look into buildLoopNest() for loop generation
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

mlir::Value getDenseMatrixFromMemRef(mlir::Location loc,
                                     mlir::ConversionPatternRewriter &rewriter,
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

    // debug
    // rewriter.create<mlir::daphne::PrintMemRef>(loc, alignedPtr, offset,
    //                                            sizes[0], sizes[1],
    //                                            strides[0], strides[1]);

    return rewriter.create<mlir::daphne::GetDenseMatrixFromMemRef>(
        loc, type, alignedPtr, offset, sizes[0], sizes[1], strides[0],
        strides[1]);
}
