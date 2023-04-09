#include <ir/daphneir/Daphne.h>

#include <iostream>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

mlir::LogicalResult mlir::daphne::GetMemRefDenseMatrix::canonicalize(
    mlir::daphne::GetMemRefDenseMatrix op, mlir::PatternRewriter &rewriter) {
    // removes unnecessary conversions of MemRef -> DM -> MemRef
    // %8 = "daphne.getDenseMatrixFromMemRef"(%intptr, ...) : (index, ...) ->
    // !daphne.Matrix<2x2xf64> %9 = "daphne.getMemRefDenseMatrix"(%8) :
    // (!daphne.Matrix<2x2xf64>) -> memref<2x2xf64>

#if 0
    std::cout << "===== GetMemRef canonicalizer =====\n";
    op->dump();
#endif

    mlir::Operation *dmNode = op->getOperand(0).getDefiningOp();

    if (!llvm::isa<mlir::daphne::GetDenseMatrixFromMemRef>(dmNode))
        return failure();

    mlir::Operation *originalMemRefOp =
        dmNode->getPrevNode()->getOperand(0).getDefiningOp();
    op.replaceAllUsesWith(originalMemRefOp);

    rewriter.eraseOp(op);
    if (dmNode->getUsers().empty()) rewriter.eraseOp(dmNode);
    return mlir::success();
}

mlir::LogicalResult mlir::daphne::GetDenseMatrixFromMemRef::canonicalize(
    mlir::daphne::GetDenseMatrixFromMemRef op,
    mlir::PatternRewriter &rewriter) {
    return mlir::success();
}

