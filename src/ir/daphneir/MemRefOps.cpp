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

#include <ir/daphneir/Daphne.h>

#include <iostream>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

mlir::LogicalResult mlir::daphne::ConvertDenseMatrixToMemRef::canonicalize(
    mlir::daphne::ConvertDenseMatrixToMemRef op,
    mlir::PatternRewriter &rewriter) {
    // removes unnecessary conversions of MemRef -> DM -> MemRef
    mlir::Operation *dmNode = op->getOperand(0).getDefiningOp();

    if (!llvm::isa<mlir::daphne::ConvertMemRefToDenseMatrix>(dmNode))
        return failure();

    mlir::Operation *originalMemRefOp =
        dmNode->getPrevNode()->getOperand(0).getDefiningOp();
    op.replaceAllUsesWith(originalMemRefOp);

    rewriter.eraseOp(op);
    if (dmNode->getUsers().empty()) rewriter.eraseOp(dmNode);

    return mlir::success();
}

mlir::LogicalResult mlir::daphne::ConvertMemRefToDenseMatrix::canonicalize(
    mlir::daphne::ConvertMemRefToDenseMatrix op,
    mlir::PatternRewriter &rewriter) {
    mlir::Operation *extractPtr = op->getPrevNode();
    auto srcMemRef = extractPtr->getOperand(0).getDefiningOp();
    extractPtr->moveAfter(srcMemRef);
    op->moveAfter(extractPtr);

    return mlir::success();
}

