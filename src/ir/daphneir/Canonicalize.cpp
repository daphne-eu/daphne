/*
 * Copyright 2024 The DAPHNE Consortium
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

#include "ir/daphneir/Daphne.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include <compiler/utils/CompilerUtils.h>

mlir::LogicalResult mlir::daphne::VectorizedPipelineOp::canonicalize(mlir::daphne::VectorizedPipelineOp op,
                                                                     mlir::PatternRewriter &rewriter) {
    // // Find duplicate inputs
    std::vector<Attribute> vSplitsAttrs;
    for (auto &split : op.getSplits())
        vSplitsAttrs.push_back(split);
    auto currentSize = op.getInputs().size();

    DenseMap<Value, size_t> inputMap;

    for (size_t i = 0; i < currentSize; i++) {
        const auto &input = op.getInputs()[i];
        const auto &split = vSplitsAttrs[i].cast<daphne::VectorSplitAttr>().getValue();

        if (inputMap.count(input) == 0) {
            inputMap[input] = i;
        } else {
            size_t j = inputMap[input];
            if (op.getSplits()[j].cast<daphne::VectorSplitAttr>().getValue() == split) {
                op.getBody().getArgument(i).replaceAllUsesWith(op.getBody().getArgument(j));
                op.getBody().eraseArgument(i);
                op.getInputsMutable().erase(i);
                vSplitsAttrs.erase(vSplitsAttrs.begin() + i);
                currentSize--;
                i--;
            }
        }
    }

    std::vector<Value> resultsToReplace;
    std::vector<Value> outRows;
    std::vector<Value> outCols;
    std::vector<Attribute> vCombineAttrs;

    llvm::BitVector eraseIxs;
    eraseIxs.resize(op.getNumResults());
    for (auto result : op.getResults()) {
        auto resultIx = result.getResultNumber();
        if (result.use_empty()) {
            // remove
            eraseIxs.set(resultIx);
        } else {
            resultsToReplace.push_back(result);
            outRows.push_back(op.getOutRows()[resultIx]);
            outCols.push_back(op.getOutCols()[resultIx]);
            vCombineAttrs.push_back(op.getCombines()[resultIx]);
        }
    }
    op.getBody().front().getTerminator()->eraseOperands(eraseIxs);
    if (!op.getCuda().getBlocks().empty())
        op.getCuda().front().getTerminator()->eraseOperands(eraseIxs);

    if (resultsToReplace.size() == op->getNumResults() && op.getSplits().size() == vSplitsAttrs.size()) {
        return failure();
    }
    auto pipelineOp = rewriter.create<daphne::VectorizedPipelineOp>(
        op.getLoc(), ValueRange(resultsToReplace).getTypes(), op.getInputs(), outRows, outCols,
        rewriter.getArrayAttr(vSplitsAttrs), rewriter.getArrayAttr(vCombineAttrs), op.getCtx());
    pipelineOp.getBody().takeBody(op.getBody());
    if (!op.getCuda().getBlocks().empty())
        pipelineOp.getCuda().takeBody(op.getCuda());
    for (auto e : llvm::enumerate(resultsToReplace)) {
        auto resultToReplace = e.value();
        auto i = e.index();
        resultToReplace.replaceAllUsesWith(pipelineOp.getResult(i));
    }
    op.erase();
    return success();
}

/**
 * @brief Transposition-aware matrix multiplication
 * Identifies if an input to a MatMulOp is the result of a TransposeOp; Rewrites
 * the Operation, passing transposition info as a flag, instead of transposing
 * the matrix before multiplication
 */
mlir::LogicalResult mlir::daphne::MatMulOp::canonicalize(mlir::daphne::MatMulOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    mlir::Value transa = op.getTransa();
    mlir::Value transb = op.getTransb();

    // TODO If transa or transb are not constant, we cannot continue on the
    // respective side; we cannot just assume false then.
    bool ta = CompilerUtils::constantOrDefault<bool>(transa, false);
    bool tb = CompilerUtils::constantOrDefault<bool>(transb, false);

    // TODO Turn on the transposition-awareness for the left-hand-side argument
    // again (see #447). mlir::daphne::TransposeOp lhsTransposeOp =
    // lhs.getDefiningOp<mlir::daphne::TransposeOp>();
    mlir::daphne::TransposeOp rhsTransposeOp = rhs.getDefiningOp<mlir::daphne::TransposeOp>();

    // if (!lhsTransposeOp && !rhsTransposeOp){
    if (!rhsTransposeOp) {
        return mlir::failure();
    }

    // ToDo: This check prevents merging transpose into matrix multiplication
    // because that is not yet supported by our
    //   sparse kernels.
    // ToDo: bring user config here for sparsity threshold or properly use
    // MatrixRepresentation
    if (auto t = rhs.getType().dyn_cast<mlir::daphne::MatrixType>()) {
        auto sparsity = t.getSparsity();
        if (sparsity < 0.25)
            return mlir::failure();
    }

#if 0
    // TODO Adapt PhyOperatorSelectionPass once this code is turned on again.
    if(lhsTransposeOp) {
        lhs = lhsTransposeOp.getArg();
        ta = !ta;
    }
#endif
    if (rhsTransposeOp) {
        rhs = rhsTransposeOp.getArg();
        tb = !tb;
    }

    rewriter.replaceOpWithNewOp<mlir::daphne::MatMulOp>(
        op, op.getType(), lhs, rhs,
        static_cast<mlir::Value>(rewriter.create<mlir::daphne::ConstantOp>(transa.getLoc(), ta)),
        static_cast<mlir::Value>(rewriter.create<mlir::daphne::ConstantOp>(transb.getLoc(), tb)));
    return mlir::success();
}

/**
 * @brief Replaces NumRowsOp by a constant, if the #rows of the input is known
 * (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumRowsOp::canonicalize(mlir::daphne::NumRowsOp op, PatternRewriter &rewriter) {
    ssize_t numRows = -1;

    mlir::Type inTy = op.getArg().getType();
    if (auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numRows = t.getNumRows();
    else if (auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numRows = t.getNumRows();

    if (numRows != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(op, rewriter.getIndexType(),
                                                              rewriter.getIndexAttr(numRows));
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces NumColsOp by a constant, if the #cols of the input is known
 * (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumColsOp::canonicalize(mlir::daphne::NumColsOp op, PatternRewriter &rewriter) {
    ssize_t numCols = -1;

    mlir::Type inTy = op.getArg().getType();
    if (auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        numCols = t.getNumCols();
    else if (auto t = inTy.dyn_cast<mlir::daphne::FrameType>())
        numCols = t.getNumCols();

    if (numCols != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(op, rewriter.getIndexType(),
                                                              rewriter.getIndexAttr(numCols));
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces NumCellsOp by a constant, if the #rows and #cols of the
 * input is known (e.g., due to shape inference).
 */
mlir::LogicalResult mlir::daphne::NumCellsOp::canonicalize(mlir::daphne::NumCellsOp op, PatternRewriter &rewriter) {
    ssize_t numRows = -1;
    ssize_t numCols = -1;

    mlir::Type inTy = op.getArg().getType();
    if (auto t = inTy.dyn_cast<mlir::daphne::MatrixType>()) {
        numRows = t.getNumRows();
        numCols = t.getNumCols();
    } else if (auto t = inTy.dyn_cast<mlir::daphne::FrameType>()) {
        numRows = t.getNumRows();
        numCols = t.getNumCols();
    }

    if (numRows != -1 && numCols != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(op, rewriter.getIndexType(),
                                                              rewriter.getIndexAttr(numRows * numCols));
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces SparsityOp by a constant, if the sparsity of the input is
 * known (e.g., due to sparsity inference).
 */
mlir::LogicalResult mlir::daphne::SparsityOp::canonicalize(mlir::daphne::SparsityOp op, PatternRewriter &rewriter) {
    double sparsity = -1.0;

    mlir::Type inTy = op.getArg().getType();
    if (auto t = inTy.dyn_cast<mlir::daphne::MatrixType>())
        sparsity = t.getSparsity();

    if (sparsity != -1) {
        rewriter.replaceOpWithNewOp<mlir::daphne::ConstantOp>(op, sparsity);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces (1) `a + b` by `a concat b`, if `a` or `b` is a string,
 * and (2) `a + X` by `X + a` (`a` scalar, `X` matrix/frame).
 *
 * (1) is important, since we use the `+`-operator for both addition and
 * string concatenation in DaphneDSL, while the types of the operands might be
 * known only after type inference.
 *
 * (2) is important, since our kernels for elementwise binary operations only
 * support scalars as the right-hand-side operand so far (see #203).
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwAddOp::canonicalize(mlir::daphne::EwAddOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    // This will check for the fill operation on the left hand side to push down the arithmetic inside
    // of it
    auto lhsFill = lhs.getDefiningOp<mlir::daphne::FillOp>();
    if (lhsFill) {
        auto lhsOp = lhs.getDefiningOp();
        auto fillValue = lhsOp->getOperand(0);
        auto width = lhsOp->getOperand(1);
        auto height = lhsOp->getOperand(2);
        const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
        if (rhsIsSca) {
            mlir::daphne::EwAddOp newAdd = rewriter.create<mlir::daphne::EwAddOp>(op.getLoc(), fillValue, rhs);
            mlir::daphne::FillOp newFill =
                rewriter.create<mlir::daphne::FillOp>(op.getLoc(), op.getResult().getType(), newAdd, width, height);
            rewriter.replaceOp(op, {newFill});
            // releasing the old fill operation
            rewriter.eraseOp(lhsOp);
            return mlir::success();
        }
    }

    const bool lhsIsStr = llvm::isa<mlir::daphne::StringType>(lhs.getType());
    const bool rhsIsStr = llvm::isa<mlir::daphne::StringType>(rhs.getType());
    if (lhsIsStr || rhsIsStr) {
        mlir::Type strTy = mlir::daphne::StringType::get(rewriter.getContext());
        if (!lhsIsStr) {
            const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
            if (!lhsIsSca) {
                const bool lhsIsMatStr =
                    llvm::isa<mlir::daphne::StringType>(lhs.getType().dyn_cast<daphne::MatrixType>().getElementType());
                if (lhsIsMatStr) {
                    rewriter.replaceOpWithNewOp<mlir::daphne::EwConcatOp>(op, op.getResult().getType(), lhs, rhs);
                    return mlir::success();
                }
            } else
                lhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, lhs);
        }
        if (!rhsIsStr)
            rhs = rewriter.create<mlir::daphne::CastOp>(op.getLoc(), strTy, rhs);
        rewriter.replaceOpWithNewOp<mlir::daphne::EwConcatOp>(op, op.getResult().getType(), lhs, rhs);
        return mlir::success();
    } else {
        const bool lhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(lhs.getType());
        const bool rhsIsSca = !llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(rhs.getType());
        if (lhsIsSca && !rhsIsSca) {
            rewriter.replaceOpWithNewOp<mlir::daphne::EwAddOp>(op, op.getResult().getType(), rhs, lhs);
            return mlir::success();
        } else if (!lhsIsSca && !rhsIsSca) {
            const bool lhsIsMatStr =
                llvm::isa<mlir::daphne::StringType>(lhs.getType().dyn_cast<daphne::MatrixType>().getElementType());
            const bool rhsIsMatStr =
                llvm::isa<mlir::daphne::StringType>(rhs.getType().dyn_cast<daphne::MatrixType>().getElementType());
            if (lhsIsMatStr && rhsIsMatStr) {
                rewriter.replaceOpWithNewOp<mlir::daphne::EwConcatOp>(op, op.getResult().getType(), lhs, rhs);
                return mlir::success();
            }
        }
        return mlir::failure();
    }
}

/**
 * @brief Replaces `a - X` by `(X * -1) + a` (`a` scalar, `X` matrix/frame).
 *
 * This is important, since our kernels for elementwise binary operations only
 * support scalars as the right-hand-side operand so far (see #203).
 *
 * As a downside, an additional operation and intermediate result is introduced.
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwSubOp::canonicalize(mlir::daphne::EwSubOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = CompilerUtils::isScaType(lhs.getType());
    const bool rhsIsSca = CompilerUtils::isScaType(rhs.getType());
    if (lhsIsSca && !rhsIsSca) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwAddOp>(
            op, op.getResult().getType(),
            rewriter.create<mlir::daphne::EwMulOp>(
                op->getLoc(),
                mlir::daphne::UnknownType::get(op->getContext()), // to be inferred
                rhs, rewriter.create<mlir::daphne::ConstantOp>(op->getLoc(), int64_t(-1))),
            lhs);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `a * X` by `X * a` (`a` scalar, `X` matrix/frame).
 *
 * This is important, since our kernels for elementwise binary operations only
 * support scalars as the right-hand-side operand so far (see #203).
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwMulOp::canonicalize(mlir::daphne::EwMulOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = CompilerUtils::isScaType(lhs.getType());
    const bool rhsIsSca = CompilerUtils::isScaType(rhs.getType());
    if (lhsIsSca && !rhsIsSca) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwMulOp>(op, op.getResult().getType(), rhs, lhs);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `a / X` by `(X ^ -1) * a` (`a` scalar, `X` matrix/frame),
 * if `X` has a floating-point value type.
 *
 * This is important, since our kernels for elementwise binary operations only
 * support scalars as the right-hand-side operand so far (see #203).
 *
 * As a downside, an additional operation and intermediate result is introduced.
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwDivOp::canonicalize(mlir::daphne::EwDivOp op, PatternRewriter &rewriter) {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    const bool lhsIsSca = CompilerUtils::isScaType(lhs.getType());
    const bool rhsIsSca = CompilerUtils::isScaType(rhs.getType());
    const bool rhsIsFP = llvm::isa<mlir::FloatType>(CompilerUtils::getValueType(rhs.getType()));
    if (lhsIsSca && !rhsIsSca && rhsIsFP) {
        rewriter.replaceOpWithNewOp<mlir::daphne::EwMulOp>(
            op, op.getResult().getType(),
            rewriter.create<mlir::daphne::EwPowOp>(op->getLoc(),
                                                   mlir::daphne::UnknownType::get(op->getContext()), // to be inferred
                                                   rhs,
                                                   rewriter.create<mlir::daphne::ConstantOp>(op->getLoc(), double(-1))),
            lhs);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces a `DistributeOp` by a `DistributedReadOp`, if its input
 * value (a) is defined by a `ReadOp`, and (b) is not used elsewhere.
 * @param context
 */
struct SimplifyDistributeRead : public mlir::OpRewritePattern<mlir::daphne::DistributeOp> {
    SimplifyDistributeRead(mlir::MLIRContext *context) : OpRewritePattern<mlir::daphne::DistributeOp>(context, 1) {
        //
    }

    mlir::LogicalResult matchAndRewrite(mlir::daphne::DistributeOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::daphne::ReadOp readOp = op.getMat().getDefiningOp<mlir::daphne::ReadOp>();
        if (!readOp || !readOp.getOperation()->hasOneUse())
            return mlir::failure();
        rewriter.replaceOp(op, {rewriter.create<mlir::daphne::DistributedReadOp>(readOp.getLoc(), op.getType(),
                                                                                 readOp.getFileName())});
        // TODO Instead of erasing the ReadOp here, the compiler should
        // generally remove unused SSA values. Then, we might even drop the
        // hasOneUse requirement above.
        rewriter.eraseOp(readOp);
        return mlir::success();
    }
};

void mlir::daphne::DistributeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<SimplifyDistributeRead>(context);
}

mlir::LogicalResult mlir::daphne::CondOp::canonicalize(mlir::daphne::CondOp op, mlir::PatternRewriter &rewriter) {
    mlir::Value cond = op.getCond();
    if (CompilerUtils::hasObjType(cond) || llvm::isa<mlir::daphne::UnknownType>(cond.getType()))
        // If the condition is not a scalar, we cannot rewrite the operation
        // here.
        return mlir::failure();
    if (CompilerUtils::hasScaType(cond)) {
        // If the condition is a scalar, we rewrite the operation to an
        // if-then-else construct using the SCF dialect.

        mlir::Location loc = op.getLoc();

        // Ensure that the condition is a boolean.
        if (!cond.getType().isSignlessInteger(1))
            cond = rewriter.create<mlir::daphne::CastOp>(loc, rewriter.getI1Type(), cond);

        mlir::Block thenBlock;
        mlir::Block elseBlock;
        mlir::Value thenVal = op.getThenVal();
        mlir::Value elseVal = op.getElseVal();

        // Get rid of frame column labels, since they interfere with the type
        // comparison (see #485).
        if (auto thenFrmTy = thenVal.getType().dyn_cast<daphne::FrameType>())
            if (thenFrmTy.getLabels() != nullptr)
                thenVal = rewriter.create<mlir::daphne::CastOp>(loc, thenFrmTy.withLabels(nullptr), thenVal);
        if (auto elseFrmTy = elseVal.getType().dyn_cast<daphne::FrameType>())
            if (elseFrmTy.getLabels() != nullptr)
                elseVal = rewriter.create<mlir::daphne::CastOp>(loc, elseFrmTy.withLabels(nullptr), elseVal);

        // Check if the types of the then-value and the else-value are the same.
        if (thenVal.getType() != elseVal.getType()) {
            if (llvm::isa<daphne::UnknownType>(thenVal.getType()) || llvm::isa<daphne::UnknownType>(elseVal.getType()))
                // If one of them is unknown, we abort the rewrite (but this is
                // not an error). The type may become known later, this rewrite
                // will be triggered again.
                return mlir::failure();
            else
                // If both types are known, but different, this is an error.
                // TODO We could try to cast the types.
                throw ErrorHandler::compilerError(op, "CanonicalizerPass (mlir::daphne::CondOp)",
                                                  "the then/else-values of CondOp must have the same value "
                                                  "type");
        }

        {
            // Save the insertion point (automatically restored at the end of
            // the block).
            PatternRewriter::InsertionGuard insertGuard(rewriter);

            // TODO The current implementation only makes sure that the correct
            // value is returned, but the operations calculating the
            // then/else-values are still outside the if-then-else and will
            // always both be executed (unless, e.g., the entire branching can
            // be elimitated). This could be good (e.g., if the then/else-values
            // have common subexpressions with other code) or bad (e.g., if they
            // are expensive to compute). See #486.

            // Create yield-operations in both branches.
            rewriter.setInsertionPointToEnd(&thenBlock);
            rewriter.create<mlir::scf::YieldOp>(loc, thenVal);
            rewriter.setInsertionPointToEnd(&elseBlock);
            rewriter.create<mlir::scf::YieldOp>(loc, elseVal);
        }

        // Helper functions to move the operations in the two blocks created
        // above into the actual branches of the if-operation.
        auto insertThenBlockDo = [&](mlir::OpBuilder &nested, mlir::Location loc) {
            nested.getBlock()->getOperations().splice(nested.getBlock()->end(), thenBlock.getOperations());
        };
        auto insertElseBlockDo = [&](mlir::OpBuilder &nested, mlir::Location loc) {
            nested.getBlock()->getOperations().splice(nested.getBlock()->end(), elseBlock.getOperations());
        };

        // Replace the daphne::CondOp by an scf::IfOp.
        rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(op, cond, insertThenBlockDo, insertElseBlockDo);

        return mlir::success();
    }

    std::stringstream s;
    s << "CondOp::canonicalize(): the condition has neither a supported data type nor a supported value type";
    throw std::runtime_error(s.str());
}

mlir::LogicalResult mlir::daphne::ConvertDenseMatrixToMemRef::canonicalize(mlir::daphne::ConvertDenseMatrixToMemRef op,
                                                                           mlir::PatternRewriter &rewriter) {
    // removes unnecessary conversions of MemRef -> DM -> MemRef
    mlir::Operation *dmNode = op->getOperand(0).getDefiningOp();

    if (!llvm::isa<mlir::daphne::ConvertMemRefToDenseMatrix>(dmNode))
        return failure();

    mlir::Operation *originalMemRefOp = dmNode->getPrevNode()->getOperand(0).getDefiningOp();
    op.replaceAllUsesWith(originalMemRefOp);

    rewriter.eraseOp(op);
    if (dmNode->getUsers().empty())
        rewriter.eraseOp(dmNode);

    return mlir::success();
}

mlir::LogicalResult mlir::daphne::ConvertMemRefToDenseMatrix::canonicalize(mlir::daphne::ConvertMemRefToDenseMatrix op,
                                                                           mlir::PatternRewriter &rewriter) {
    mlir::Operation *extractPtr = op->getPrevNode();
    auto srcMemRef = extractPtr->getOperand(0).getDefiningOp();
    extractPtr->moveAfter(srcMemRef);
    op->moveAfter(extractPtr);

    return mlir::success();
}

/**
 * @brief Replaces `floor(a)` with `a` if `a` is an integer
 * or a matrix of integers.
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwFloorOp::canonicalize(mlir::daphne::EwFloorOp op, mlir::PatternRewriter &rewriter) {
    mlir::Value operand = op.getOperand();
    auto matrix = operand.getType().dyn_cast<mlir::daphne::MatrixType>();
    mlir::Type elemType = matrix ? matrix.getElementType() : operand.getType();

    if (llvm::isa<mlir::IntegerType>(elemType)) {
        rewriter.replaceOp(op, operand);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `ceil(a)` with `a` if `a` is an integer
 * or a matrix of integers.
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwCeilOp::canonicalize(mlir::daphne::EwCeilOp op, mlir::PatternRewriter &rewriter) {
    mlir::Value operand = op.getOperand();
    auto matrix = operand.getType().dyn_cast<mlir::daphne::MatrixType>();
    mlir::Type elemType = matrix ? matrix.getElementType() : operand.getType();

    if (llvm::isa<mlir::IntegerType>(elemType)) {
        rewriter.replaceOp(op, operand);
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Replaces `round(a)` with `a` if `a` is an integer
 * or a matrix of integers.
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwRoundOp::canonicalize(mlir::daphne::EwRoundOp op, mlir::PatternRewriter &rewriter) {
    mlir::Value operand = op.getOperand();
    auto matrix = operand.getType().dyn_cast<mlir::daphne::MatrixType>();
    mlir::Type elemType = matrix ? matrix.getElementType() : operand.getType();

    if (llvm::isa<mlir::IntegerType>(elemType)) {
        rewriter.replaceOp(op, operand);
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult mlir::daphne::RenameOp::canonicalize(mlir::daphne::RenameOp op, mlir::PatternRewriter &rewriter) {
    // Replace the RenameOp by its argument, since we only need
    // this operation during DaphneDSL parsing.
    rewriter.replaceOp(op, op.getArg());
    return mlir::success();
}

/**
 * @brief Replaces `--a` by `a` (`a` scalar).
 *
 * @param op
 * @param rewriter
 * @return
 */
mlir::LogicalResult mlir::daphne::EwMinusOp::canonicalize(mlir::daphne::EwMinusOp op, PatternRewriter &rewriter) {
    if (auto innerOp = op.getOperand().getDefiningOp<mlir::daphne::EwMinusOp>()) {
        rewriter.replaceOp(op, innerOp.getOperand());
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Eliminates redundant conversions of a position list to a bitmap and back to a position list.
 *
 * This pattern frequently occurs during lowering to columnar operations. This simplification rewrite avoids the
 * unnecessary creation of the intermediate bitmap.
 */
mlir::LogicalResult mlir::daphne::ConvertBitmapToPosListOp::canonicalize(mlir::daphne::ConvertBitmapToPosListOp bmplcOp,
                                                                         PatternRewriter &rewriter) {
    if (auto plbmcOp = bmplcOp.getArg().getDefiningOp<mlir::daphne::ConvertPosListToBitmapOp>()) {
        // The ConvertPosListToBitmapOp has a second argument for the number of rows (bitmap size). No matter what this
        // size is, we always get back the original position list. The only exception would be if the bitmap size is
        // less than the greatest position in the original position list (information loss or error during conversion
        // from position list to bitmap). However, this case is not relevant to us at the moment.
        rewriter.replaceOp(bmplcOp, plbmcOp.getArg());
        return mlir::success();
    }
    return mlir::failure();
}

/**
 * @brief Simplifies the extraction of a single column (by its label) from a frame in various situations.
 */
mlir::LogicalResult mlir::daphne::ExtractColOp::canonicalize(mlir::daphne::ExtractColOp ecOp,
                                                             PatternRewriter &rewriter) {
    Value src = ecOp.getSource();
    Value sel = ecOp.getSelectedCols();
    if (auto srcFrmTy = src.getType().dyn_cast<mlir::daphne::FrameType>()) {
        if (sel.getType().isa<mlir::daphne::StringType>()) {
            if (srcFrmTy.getNumCols() == 1) {
                // Eliminate the extraction of a single column (by its label) from a frame with a single column which
                // has exactly that label.

                if (std::vector<std::string> *labels = srcFrmTy.getLabels()) {
                    std::pair<bool, std::string> selConst = CompilerUtils::isConstant<std::string>(sel);
                    if (selConst.first && selConst.second == (*labels)[0]) {
                        rewriter.replaceOp(ecOp, src);
                        return mlir::success();
                    }
                }
            } else if (auto cfOp = src.getDefiningOp<mlir::daphne::CreateFrameOp>()) {
                // Replace the extraction of a single column (by its label) from the result of a CreateFrameOp, by the
                // respective input column of the CreateFrameOp. This simplification rewrite can help us avoid the
                // unnecessary creation of frames when we are interested only in certain columns later on. It can lead
                // to the elimination of the operations creating later unused input columns of the CreateFrameOp.

                // CreateFrameOp always has an even number of arguments. The first half are the columns and the second
                // half are the labels.
                const size_t cfNumCols = cfOp->getNumOperands() / 2;
                // Search for the column with the specified label.
                for (size_t i = 0; i < cfNumCols; i++) {
                    Value label = cfOp->getOperand(cfNumCols + i);
                    if (label == sel) {
                        // We need to insert an additional cast of the CreateFrameOp's input column to the result type
                        // of the ExtractColOp, because usually, the arguments to CreateFrameOp are single-column
                        // *matrices*, while the result of ExtractColOp on a frame is a single-column *frame*. Such
                        // additional casts will often be eliminated through the canonicalization of CastOp later.
                        Type resTy = ecOp.getResult().getType();
                        if (auto resFrmTy = resTy.dyn_cast<mlir::daphne::FrameType>()) {
                            // Reset the labels to unknown, because TODO
                            Value casted = rewriter
                                               .create<mlir::daphne::CastOp>(
                                                   ecOp.getLoc(), resFrmTy.withLabels(nullptr), cfOp->getOperand(i))
                                               .getResult();
                            // If the result of the ExtractColOp is a (single-column) frame (typically the case), we
                            // must make sure that it gets the right column label after the rewrite. The label is an
                            // argument to the CreateFrameOp and might not be known as a compile-time constant at the
                            // point in time when this rewrite happens (the label could be the result of a complex
                            // string expression, which is resolved later during compile-time through constant folding
                            // or only at run-time). Thus, we additionally insert a SetColLabelsOp which reuses exactly
                            // the same input as the CreateFrameOp for the label.
                            Value labeled =
                                rewriter.create<mlir::daphne::SetColLabelsOp>(ecOp.getLoc(), resTy, casted, label)
                                    .getResult();
                            rewriter.replaceOp(ecOp, labeled);
                            return success();
                        }
                    }
                }
            } else if (auto cbOp = src.getDefiningOp<mlir::daphne::ColBindOp>()) {
                // Push down the extraction of a single column (by its label) from the result of a ColBindOp to the
                // argument of the ColBindOp that has the column with the desired label.

                std::pair<bool, std::string> selConst = CompilerUtils::isConstant<std::string>(sel);
                if (selConst.first) {
                    auto tryOneArg = [&](Value arg) {
                        if (auto argFrmTy = arg.getType().dyn_cast<mlir::daphne::FrameType>())
                            if (argFrmTy.getLabels() != nullptr)
                                for (std::string label : *(argFrmTy.getLabels()))
                                    if (label == selConst.second) {
                                        rewriter.replaceOpWithNewOp<mlir::daphne::ExtractColOp>(ecOp, ecOp.getType(),
                                                                                                arg, sel);
                                        return true;
                                    }
                        return false;
                    };
                    if (tryOneArg(cbOp.getLhs()) || tryOneArg(cbOp.getRhs()))
                        return mlir::success();
                }
            } else if (auto ecOp2 = src.getDefiningOp<mlir::daphne::ExtractColOp>()) {
                // Eliminate two subsequent extractions of a single column (by its label) with the same label.

                if (ecOp2.getSelectedCols() == sel) {
                    rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(ecOp, ecOp.getResult().getType(),
                                                                      ecOp2.getResult());
                    return mlir::success();
                }
            }
        }
    }
    return mlir::failure();
}

/**
 * @brief Simplifies various patterns of ops that end with a CastOp.
 *
 * Simple examples include the elimination of trivial casts (casting from A to A) and the simplification of chains of
 * casts (e.g., casting from A to B to C becomes casting from A to C, in case there is no information loss). Besides
 * that, there are patterns that by-pass ops that create the input of a CastOp and patterns that by-pass the CastOp
 * itself.
 */
mlir::LogicalResult mlir::daphne::CastOp::canonicalize(mlir::daphne::CastOp cOp, PatternRewriter &rewriter) {
    // TODO Maybe skip property casts, or separate them, combine multiple property casts.

    // Replace cast "a -> a".
    if (cOp.isTrivialCast()) {
        rewriter.replaceOp(cOp, cOp.getArg());
        return mlir::success();
    }

    // Replace cast "a -> b -> c" to cast "a -> c", if the cast "a -> b" does not lose information, because if "b"
    // contains the same information as "a", we could directly cast from "a" to "c".
    // It does not matter if "b -> c" may lose information, because "b" does not have more information than "a".
    if (auto cOp0 = cOp.getArg().getDefiningOp<mlir::daphne::CastOp>()) {
        if (!cOp0.mightLoseInformation()) {
            rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(cOp, cOp.getRes().getType(), cOp0.getArg());
            return mlir::success();
        }
    }

    // Bypass operations manipulating frame column labels in case their result is casted to a data type that does not
    // support column labels (matrix/column). This is sound, since the column label information from the frame would be
    // gone after the cast anyway.
    if (cOp.getArg().getType().isa<mlir::daphne::FrameType>() &&
        cOp.getRes().getType().isa<mlir::daphne::MatrixType, mlir::daphne::ColumnType>()) {
        if (auto sclOp = cOp.getArg().getDefiningOp<mlir::daphne::SetColLabelsOp>()) {
            rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(cOp, cOp.getRes().getType(), sclOp.getArg());
            return mlir::success();
        }
        if (auto sclpOp = cOp.getArg().getDefiningOp<mlir::daphne::SetColLabelsPrefixOp>()) {
            rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(cOp, cOp.getRes().getType(), sclpOp.getArg());
            return mlir::success();
        }
    }

    // Bypass CreateFrameOp (with a single input matrix) followed by CastOp to the same matrix type as the
    // CreateFrameOp's input matrix. In such cases, the result of the CastOp is simply the argument of the
    // CreateFrameOp.
    // TODO check if the matrix has a single column, otherwise, we would bypass the check in createframe
    if (cOp.getArg().getType().isa<mlir::daphne::FrameType>() && cOp.getRes().getType().isa<mlir::daphne::MatrixType>())
        if (auto cfOp = cOp.getArg().getDefiningOp<mlir::daphne::CreateFrameOp>())
            if (cfOp.getCols().size() == 1 && cfOp.getCols()[0].getType() == cOp.getRes().getType()) {
                rewriter.replaceOp(cOp, cfOp.getCols()[0]);
                return mlir::success();
            }

    return mlir::failure();
}

/**
 * @brief Eliminates a SetColLabelsOp if the input frame already has the new column labels.
 */
mlir::LogicalResult mlir::daphne::SetColLabelsOp::canonicalize(mlir::daphne::SetColLabelsOp sclOp,
                                                               PatternRewriter &rewriter) {
    if (auto argFrmTy = sclOp.getArg().getType().dyn_cast<mlir::daphne::FrameType>()) { // if the arg is a frame
        if (std::vector<std::string> *argLabels = argFrmTy.getLabels()) {               // if the arg's labels are known
            // Compare the arg's labels with the new labels.
            mlir::ValueRange newLabels = sclOp.getLabels();
            if (argLabels->size() != newLabels.size())
                return mlir::failure();
            for (size_t i = 0; i < newLabels.size(); i++) {
                std::pair<bool, std::string> labelConst = CompilerUtils::isConstant<std::string>(newLabels[i]);
                if (!labelConst.first ||
                    labelConst.second != (*argLabels)[i]) // the new label is not known or differs from the arg label
                    return mlir::failure();
            }
            // The arg frame already has the new labels, so we can replace this SetColLabelsOp by its arg.
            rewriter.replaceOp(sclOp, sclOp.getArg());
            return mlir::success();
        }
    }
    return mlir::failure();
}
