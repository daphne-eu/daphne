#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <iostream>

#include "llvm/Support/Casting.h"
#include "mlir/Support/LogicalResult.h"
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"

mlir::LogicalResult mlir::daphne::GetMemRefDenseMatrix::canonicalize(
    mlir::daphne::GetMemRefDenseMatrix op, mlir::PatternRewriter &rewriter) {
    // remove structures like this one:
    // %8 = "daphne.getDenseMatrixFromMemRef"(%intptr, %offset, %sizes#0,
    // %sizes#1, %strides#0, %strides#1) : (index, index, index, index, index,
    // index) -> !daphne.Matrix<2x2xf64> %9 = "daphne.getMemRefDenseMatrix"(%8)
    // : (!daphne.Matrix<2x2xf64>) -> memref<2x2xf64> instead of converting
    // twice we remove conversions and reuse original memref

#if 0
    std::cout << "===== GetMemRef canonicalizer =====\n";
    op->dump();
#endif
    mlir::Operation *prevNode = op->getPrevNode();
    if (!llvm::isa<mlir::daphne::GetDenseMatrixFromMemRef>(prevNode)) {
        return failure();
    }

    mlir::Operation *originalMemRefOp = prevNode->getPrevNode()->getOperand(0).getDefiningOp();

    op.replaceAllUsesWith(originalMemRefOp);

    rewriter.eraseOp(op);
    rewriter.eraseOp(prevNode);

    return mlir::success();
#if 0

    std::cout << "op memref:\n" << std::endl;
    op->getOpOperand(0).get().dump();
    std::cout << "\n";

    std::cout << "original memref:\n" << std::endl;
    originalMemRefOp->dump();
    // mlir::Value memRef = prevNode->getPrevNode()->getOpOperand(0).get();
    auto users = op->getUsers();
    std::cout << "users:\n" << std::endl;
    for (auto user : users) {
        user->dump();
    }
#endif
}

mlir::LogicalResult mlir::daphne::GetDenseMatrixFromMemRef::canonicalize(
    mlir::daphne::GetDenseMatrixFromMemRef op,
    mlir::PatternRewriter &rewriter) {
    return mlir::success();
}

