/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/DaphneInferFrameLabelsOpInterface.h>
#include <ir/daphneir/Passes.h>

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <memory>
#include <string>
#include <utility>

using namespace mlir;

namespace {

// ********************************************************************************
// Helper Functions
// ********************************************************************************

/**
 * @brief Applies a columnar projection (`ColProjectOp`) to all columns of the given frame.
 *
 * @param rewriter a `PatternRewriter`
 * @param loc the location to use for newly created ops
 * @param argFrm the frame whose columns shall be projected
 * @param selPosLstCol the position list column to use for the columnar projection
 * @param colLabelsStr the labels of the columns
 * @param cols the projected columns (used for the result)
 * @param colLabelsVal the labels of the columns (newly created `mlir::daphne::ConstantOp`s, used for the result)
 */
void projectAllColumns(PatternRewriter &rewriter, Location loc, Value argFrm, Value selPosLstCol,
                       std::vector<std::string> colLabelsStr, std::vector<Value> &cols,
                       std::vector<Value> &colLabelsVal) {
    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);
    Type mu = daphne::MatrixType::get(rewriter.getContext(), u);

    for (std::string colLabelStr : colLabelsStr) {
        // Create a ConstantOp with the label of the column, keep it for later use in CreateFrameOp.
        Value colLabelVal = rewriter.create<daphne::ConstantOp>(loc, colLabelStr);
        colLabelsVal.push_back(colLabelVal);
        // Extract the column by its label; the outcome is a single-column frame.
        Value argColFrm = rewriter.create<daphne::ExtractColOp>(loc, u, argFrm, colLabelVal);
        // Cast the single-column frame to a column.
        Value argColCol = rewriter.create<daphne::CastOp>(loc, cu, argColFrm);
        // Apply the columnar projection; the outcome is a column.
        Value resColCol = rewriter.create<daphne::ColProjectOp>(loc, u, argColCol, selPosLstCol);
        // Cast the column to a matrix, keep it for later use in CreateFrameOp.
        Value resColMat = rewriter.create<daphne::CastOp>(loc, mu, resColCol);
        cols.push_back(resColMat);
    }
}

template <class MatCmpOp, class ColSelectCmpOp> LogicalResult replaceCompareOp(PatternRewriter &rewriter, MatCmpOp op) {
    // Get the location of the op to replace; we will use it for all newly created ops.
    Location loc = op->getLoc();

    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);
    Type mu = daphne::MatrixType::get(rewriter.getContext(), u);

    // Get the left input, which is a matrix (otherwise, op would be legal).
    Value lhsMat = op.getLhs();
    // Cast the left input to a column.
    Value lhsCol = rewriter.create<daphne::CastOp>(loc, cu, lhsMat);

    // Get the right input, which is a scalar (otherwise, op would be legal).
    Value rhsSca = op.getRhs();

    // Create the columnar comparison op. The outcome is a position list represented as a column.
    Value resPosLstCol = rewriter.create<ColSelectCmpOp>(loc, u, lhsCol, rhsSca);

    // Cast the position list to a single-column matrix.
    Value resPosLstMat = rewriter.create<daphne::CastOp>(loc, mu, resPosLstCol);

    // Convert the position list to a bit vector and replace the original comparison op by it.
    Value numRows = rewriter.create<daphne::NumRowsOp>(loc, rewriter.getIndexType(), lhsMat);
    rewriter.replaceOpWithNewOp<daphne::ConvertPosListToBitmapOp>(op, op.getRes().getType(), resPosLstMat, numRows);

    return success();
}

template <class MatLogOp, class ColSetOp> LogicalResult replaceLogicalOp(PatternRewriter &rewriter, MatLogOp op) {
    // Get the location of the op to replace; we will use it for all newly created ops.
    Location loc = op->getLoc();

    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);
    Type mu = daphne::MatrixType::get(rewriter.getContext(), u);

    // Get the left and right inputs, which are matrices (otherwise, op would be legal) containing bitmaps.
    Value lhsBitmapMat = op.getLhs();
    Value rhsBitmapMat = op.getRhs();
    // Convert the left and right inputs to position lists.
    Value lhsPosLstMat = rewriter.create<daphne::ConvertBitmapToPosListOp>(loc, u, lhsBitmapMat);
    Value rhsPosLstMat = rewriter.create<daphne::ConvertBitmapToPosListOp>(loc, u, rhsBitmapMat);
    // Cast the input position lists to columns.
    Value lhsPosLstCol = rewriter.create<daphne::CastOp>(loc, cu, lhsPosLstMat);
    Value rhsPosLstCol = rewriter.create<daphne::CastOp>(loc, cu, rhsPosLstMat);

    // Create the columnar set op. The outcome is a position list represented as a column.
    Value resPosLstCol = rewriter.create<ColSetOp>(loc, u, lhsPosLstCol, rhsPosLstCol);

    // Cast the position list to a single-column matrix.
    Value resPosLstMat = rewriter.create<daphne::CastOp>(loc, mu, resPosLstCol);

    // Convert the position list to a bit vector and replace the original logical op by it.
    Value numRows = rewriter.create<daphne::NumRowsOp>(loc, rewriter.getIndexType(), lhsBitmapMat);
    rewriter.replaceOpWithNewOp<daphne::ConvertPosListToBitmapOp>(op, op.getRes().getType(), resPosLstMat, numRows);

    return success();
}

template <class FrmOp> LogicalResult replaceExtractRowOp(PatternRewriter &rewriter, FrmOp op, Value selPosLstMat) {
    // Get the location of the op to replace; we will use it for all newly created ops.
    Location loc = op->getLoc();

    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);

    // We need the information which rows to extract as a position list contained in a column.
    Value selPosLstCol = rewriter.create<daphne::CastOp>(loc, cu, selPosLstMat);

    Value src = op.getSource();

    auto srcFrmTy = src.getType().dyn_cast<daphne::FrameType>();
    // For now, we only replace the op, if it is applied to a frame.
    // TODO We could also support matrices as source.
    if (!srcFrmTy)
        return failure();

    std::vector<std::string> *srcColLabels = srcFrmTy.getLabels();
    // The column labels of the source must be known (current requirement by projectAllColumns()).
    // TODO We could relax this requirement.
    if (!srcColLabels)
        return failure();

    // Project all columns of the source separately using columnar operations.
    std::vector<Value> cols;
    std::vector<Value> colLabels;
    projectAllColumns(rewriter, loc, src, selPosLstCol, *srcColLabels, cols, colLabels);

    // Replace the op by a new CreateFrameOp consisting of the individually processed columns.
    // The result type is the same as before the rewrite. This includes the column order and column labels.
    rewriter.replaceOpWithNewOp<daphne::CreateFrameOp>(op, op.getResult().getType(), cols, colLabels);

    return success();
}

template <class MatBinOp, class ColCalcBinaryOp> LogicalResult replaceBinaryOp(PatternRewriter &rewriter, MatBinOp op) {
    // Get the location of the op to replace; we will use it for all newly created ops.
    Location loc = op->getLoc();

    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);

    // Get the left and right inputs, which are matrices (otherwise, op would be legal).
    Value lhsMat = op.getLhs();
    Value rhsMat = op.getRhs();
    // Cast the inputs to columns.
    Value lhsCol = rewriter.create<daphne::CastOp>(loc, cu, lhsMat);
    Value rhsCol = rewriter.create<daphne::CastOp>(loc, cu, rhsMat);

    // Create the columnar binary op, whose result is a column.
    Value resCol = rewriter.create<ColCalcBinaryOp>(loc, u, lhsCol, rhsCol);

    // Cast the result column to a single-column matrix and replace the original comparison op by it.
    rewriter.replaceOpWithNewOp<daphne::CastOp>(op, op.getRes().getType(), resCol);

    return success();
}

template <class MatAllAggOp, class ColAllAggOp>
LogicalResult replaceAllAggOp(PatternRewriter &rewriter, MatAllAggOp op) {
    // Get the location of the op to replace; we will use it for all newly created ops.
    Location loc = op->getLoc();

    // Some types.
    Type u = daphne::UnknownType::get(rewriter.getContext());
    Type cu = daphne::ColumnType::get(rewriter.getContext(), u);

    // Get the input, which is a matrix (otherwise, op would be legal).
    Value argMat = op.getArg();
    // Cast the input to a column.
    Value argCol = rewriter.create<daphne::CastOp>(loc, cu, argMat);

    // Create the columnar aggregation op, whose result is a single-element column.
    Value resCol = rewriter.create<ColAllAggOp>(loc, u, argCol);

    // Cast the result column to a scalar (same type of before the rewrite) and replace the original aggregation op by
    // it.
    rewriter.replaceOpWithNewOp<daphne::CastOp>(op, op.getRes().getType(), resCol);

    return success();
}

// ********************************************************************************
// Rewrite Patterns
// ********************************************************************************

struct ColumnarOpReplacement : public RewritePattern {

    ColumnarOpReplacement(MLIRContext *context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Note that all unknown data/value types we introduce when creating new ops are inferred in subsequent compiler
        // passes. We only specify the necessary type information, i.e., the result data type of casts.
        Type u = daphne::UnknownType::get(rewriter.getContext());
        Type cu = daphne::ColumnType::get(rewriter.getContext(), u);
        Type mu = daphne::MatrixType::get(rewriter.getContext(), u);

        if (auto cmpOp = llvm::dyn_cast<daphne::EwEqOp>(op)) {
            return replaceCompareOp<daphne::EwEqOp, daphne::ColSelectEqOp>(rewriter, cmpOp);
        } else if (auto cmpOp = llvm::dyn_cast<daphne::EwNeqOp>(op)) {
            return replaceCompareOp<daphne::EwNeqOp, daphne::ColSelectNeqOp>(rewriter, cmpOp);
        } else if (auto cmpOp = llvm::dyn_cast<daphne::EwGtOp>(op)) {
            return replaceCompareOp<daphne::EwGtOp, daphne::ColSelectGtOp>(rewriter, cmpOp);
        } else if (auto cmpOp = llvm::dyn_cast<daphne::EwGeOp>(op)) {
            return replaceCompareOp<daphne::EwGeOp, daphne::ColSelectGeOp>(rewriter, cmpOp);
        } else if (auto cmpOp = llvm::dyn_cast<daphne::EwLtOp>(op)) {
            return replaceCompareOp<daphne::EwLtOp, daphne::ColSelectLtOp>(rewriter, cmpOp);
        } else if (auto cmpOp = llvm::dyn_cast<daphne::EwLeOp>(op)) {
            return replaceCompareOp<daphne::EwLeOp, daphne::ColSelectLeOp>(rewriter, cmpOp);
        } else if (auto logOp = llvm::dyn_cast<daphne::EwAndOp>(op)) {
            return replaceLogicalOp<daphne::EwAndOp, daphne::ColIntersectOp>(rewriter, logOp);
        } else if (auto logOp = llvm::dyn_cast<daphne::EwOrOp>(op)) {
            return replaceLogicalOp<daphne::EwOrOp, daphne::ColMergeOp>(rewriter, logOp);
        } else if (auto frOp = llvm::dyn_cast<daphne::FilterRowOp>(op)) {
            // Get the location of the op to replace; we will use it for all newly created ops.
            Location loc = frOp->getLoc();

            // For FilterRowOp, the rows to extract are given as a bit vector (0/1 values) contained in a matrix.
            // replaceExtractRowOp() needs them as a position list contained in a matrix.
            Value selBitVecMat = frOp.getSelectedRows();
            Value selPosLstMat = rewriter.create<daphne::ConvertBitmapToPosListOp>(loc, u, selBitVecMat);

            return replaceExtractRowOp(rewriter, frOp, selPosLstMat);
        } else if (auto erOp = llvm::dyn_cast<daphne::ExtractRowOp>(op)) {
            // For ExtractRowOp, the rows to extract are given as a positions list contained in a matrix, which is what
            // replaceExtractRowOp() needs.
            Value selPosLstMat = erOp.getSelectedRows();

            return replaceExtractRowOp(rewriter, erOp, selPosLstMat);
        } else if (auto binOp = llvm::dyn_cast<daphne::EwSubOp>(op)) {
            return replaceBinaryOp<daphne::EwSubOp, daphne::ColCalcSubOp>(rewriter, binOp);
        } else if (auto binOp = llvm::dyn_cast<daphne::EwMulOp>(op)) {
            return replaceBinaryOp<daphne::EwMulOp, daphne::ColCalcMulOp>(rewriter, binOp);
        } else if (auto aggOp = llvm::dyn_cast<daphne::AllAggSumOp>(op)) {
            return replaceAllAggOp<daphne::AllAggSumOp, daphne::ColAllAggSumOp>(rewriter, aggOp);
        } else if (auto ijOp = llvm::dyn_cast<daphne::InnerJoinOp>(op)) {
            // Get the location of the op to replace; we will use it for all newly created ops.
            Location loc = ijOp->getLoc();

            Value lhs = ijOp.getLhs();     // left input frame
            Value rhs = ijOp.getRhs();     // right input frame
            Value lhsOn = ijOp.getLhsOn(); // key column label in the left input
            Value rhsOn = ijOp.getRhsOn(); // key column label in the rifht input

            auto lhsFrmTy = lhs.getType().dyn_cast<daphne::FrameType>();
            auto rhsFrmTy = rhs.getType().dyn_cast<daphne::FrameType>();
            // Both inputs must be frames.
            if (!lhsFrmTy || !rhsFrmTy)
                return failure();

            // Extract the key columns as single-column frames and cast them to columns.
            Value lhsKeysMat = rewriter.create<daphne::ExtractColOp>(loc, u, lhs, lhsOn);
            Value rhsKeysMat = rewriter.create<daphne::ExtractColOp>(loc, u, rhs, rhsOn);
            Value lhsKeysCol = rewriter.create<daphne::CastOp>(loc, cu, lhsKeysMat);
            Value rhsKeysCol = rewriter.create<daphne::CastOp>(loc, cu, rhsKeysMat);

            // Perform the columnar inner join.
            auto cjOp = rewriter.create<daphne::ColJoinOp>(loc, u, u, lhsKeysCol, rhsKeysCol, ijOp.getNumRowRes());
            Value lhsPosLstCol = cjOp.getResLhsPos();
            Value rhsPosLstCol = cjOp.getResRhsPos();

            std::vector<std::string> *lhsColLabels = lhsFrmTy.getLabels();
            std::vector<std::string> *rhsColLabels = rhsFrmTy.getLabels();
            // The column labels of both inputs must be known (current requirement by projectAllColumns()).
            // TODO We could relax this requirement.
            if (!lhsColLabels || !rhsColLabels)
                return failure();

            // Project all columns of the left and right inputs on the matching rows separately using columnar
            // operations.
            std::vector<Value> resCols;
            std::vector<Value> resColLabels;
            projectAllColumns(rewriter, loc, lhs, lhsPosLstCol, *lhsColLabels, resCols, resColLabels);
            projectAllColumns(rewriter, loc, rhs, rhsPosLstCol, *rhsColLabels, resCols, resColLabels);

            // Replace the InnerJoinOp by a new CreateFrameOp consisting of the individually processed columns.
            // The result type is the same as before the rewrite. This includes the column order and column labels.
            rewriter.replaceOpWithNewOp<daphne::CreateFrameOp>(op, ijOp.getRes().getType(), resCols, resColLabels);

            return success();
        } else if (auto sjOp = llvm::dyn_cast<daphne::SemiJoinOp>(op)) {
            // Get the location of the op to replace; we will use it for all newly created ops.
            Location loc = sjOp->getLoc();

            Value lhs = sjOp.getLhs();     // left input frame
            Value rhs = sjOp.getRhs();     // right input frame
            Value lhsOn = sjOp.getLhsOn(); // key column label in the left input
            Value rhsOn = sjOp.getRhsOn(); // key column label in the rifht input

            auto lhsFrmTy = lhs.getType().dyn_cast<daphne::FrameType>();
            auto rhsFrmTy = rhs.getType().dyn_cast<daphne::FrameType>();
            // Both inputs must be frames.
            if (!lhsFrmTy || !rhsFrmTy)
                return failure();

            // Extract the key columns as single-column frames and cast them to columns.
            Value lhsKeysMat = rewriter.create<daphne::ExtractColOp>(loc, u, lhs, lhsOn);
            Value rhsKeysMat = rewriter.create<daphne::ExtractColOp>(loc, u, rhs, rhsOn);
            Value lhsKeysCol = rewriter.create<daphne::CastOp>(loc, cu, lhsKeysMat);
            Value rhsKeysCol = rewriter.create<daphne::CastOp>(loc, cu, rhsKeysMat);

            // Perform the columnar semi join.
            Value lhsPosLstCol =
                rewriter.create<daphne::ColSemiJoinOp>(loc, u, lhsKeysCol, rhsKeysCol, sjOp.getNumRowRes());

            std::vector<std::string> *lhsColLabels = lhsFrmTy.getLabels();
            // The column labels of the left inputs must be known (current requirement by projectAllColumns()).
            // TODO We could relax this requirement.
            if (!lhsColLabels)
                return failure();

            // Project all columns of the left input on the matching rows separately using columnar operations.
            std::vector<Value> resCols;
            std::vector<Value> resColLabels;
            projectAllColumns(rewriter, loc, lhs, lhsPosLstCol, *lhsColLabels, resCols, resColLabels);

            // Replace the SemiJoinOp by two new results: a new CreateFrameOp consisting of the individually processed
            // columns and the positions of the rows in the left input that have a join partner in the right input. The
            // result types are the same as before the rewrite. This includes the column order and column labels.
            Value res = rewriter.create<daphne::CreateFrameOp>(loc, sjOp.getRes().getType(), resCols, resColLabels);
            Value lhsPosLstMat = rewriter.create<daphne::CastOp>(loc, sjOp.getLhsTids().getType(), lhsPosLstCol);
            rewriter.replaceOp(op, {res, lhsPosLstMat});

            return success();
        } else if (auto gOp = llvm::dyn_cast<daphne::GroupOp>(op)) {
            // Get the location of the op to replace; we will use it for all newly created ops.
            Location loc = gOp->getLoc();

            Value arg = gOp.getFrame();             // input frame
            ValueRange keyLabels = gOp.getKeyCol(); // labels of the columns to group on
            ValueRange aggLabels = gOp.getAggCol(); // labels of the columns to aggregate

            auto argFrmTy = arg.getType().dyn_cast<daphne::FrameType>();
            // The input must be a frame.
            if (!argFrmTy)
                return failure();

            std::vector<Value> resCols;
            std::vector<Value> resColLabels;

            // Find out the group ids and representative positions.
            std::vector<Value> keyCols;
            // Process the first key column.
            Value keyMat = rewriter.create<daphne::ExtractColOp>(loc, u, arg, keyLabels[0]);
            Value keyCol = rewriter.create<daphne::CastOp>(loc, cu, keyMat);
            keyCols.push_back(keyCol);
            auto cgfOp = rewriter.create<daphne::ColGroupFirstOp>(loc, u, u, keyCol);
            Value grpIds = cgfOp.getResGrpIds();
            Value reprPos = cgfOp.getResReprPos();
            // Process the remaining key columns.
            for (size_t i = 1; i < keyLabels.size(); i++) {
                keyMat = rewriter.create<daphne::ExtractColOp>(loc, u, arg, keyLabels[i]);
                keyCol = rewriter.create<daphne::CastOp>(loc, cu, keyMat);
                keyCols.push_back(keyCol);
                // Use the group ids from the previous ColGroupFirstOp/ColGroupNextOp.
                auto cgnOp = rewriter.create<daphne::ColGroupNextOp>(loc, u, u, keyCol, grpIds);
                grpIds = cgnOp.getResGrpIds();
                reprPos = cgnOp.getResReprPos();
            }

            // Extract the representatives from all key columns.
            for (size_t i = 0; i < keyCols.size(); i++) {
                Value keyColReprsCol = rewriter.create<daphne::ColProjectOp>(loc, u, keyCols[i], reprPos);
                Value keyColReprsMat = rewriter.create<daphne::CastOp>(loc, mu, keyColReprsCol);
                resCols.push_back(keyColReprsMat);
                resColLabels.push_back(keyLabels[i]);
            }

            // Grouped aggregation on all aggregation columns.
            Value numDistinct = rewriter.create<daphne::NumRowsOp>(loc, u, reprPos);
            for (size_t i = 0; i < aggLabels.size(); i++) {
                Value aggMat = rewriter.create<daphne::ExtractColOp>(loc, u, arg, aggLabels[i]);
                Value aggCol = rewriter.create<daphne::CastOp>(loc, cu, aggMat);
                Value aggedCol = rewriter.create<daphne::ColGrpAggSumOp>(loc, u, aggCol, grpIds, numDistinct);
                Value aggedMat = rewriter.create<daphne::CastOp>(loc, mu, aggedCol);
                resCols.push_back(aggedMat);
                // TODO Don't hardcode "SUM(", do it like the group kernel on frames does it.
                Value newLabel = rewriter.create<daphne::EwConcatOp>(
                    loc, u,
                    rewriter.create<daphne::EwConcatOp>(
                        loc, u, rewriter.create<daphne::ConstantOp>(loc, std::string("SUM(")), aggLabels[i]),
                    rewriter.create<daphne::ConstantOp>(loc, std::string(")")));
                resColLabels.push_back(newLabel);
            }

            // Replace the GroupOp by a new CreateFrameOp consisting of the individually processed columns.
            // The result type is the same as before the rewrite. This includes the column order and column labels.
            rewriter.replaceOpWithNewOp<daphne::CreateFrameOp>(op, gOp.getRes().getType(), resCols, resColLabels);

            return success();
        }

        // This should never happen (all ops to be replaced should be handled above).
        return failure();
    }
};

// ********************************************************************************
// Compiler Pass
// ********************************************************************************

/**
 * @brief Rewrites certain matrix/frame ops from linear/relational algebra to columnar ops from column algebra.
 *
 * The general idea is to identify and replace individual matrix/frame ops that (depending on the op and the types,
 * shapes, etc. of its arguments) could be expressed by columnar ops. Then, each of these ops is replaced in isolation
 * by creating casts/conversions of its arguments as needed, creating the columnar op(s), and creating casts/conversions
 * of the results as needed. In the end, the results of the rewritten DAG of operations are the same as of the replaced
 * op. After these replacements of individual ops, the IR may contain lots of redundant operations or operations
 * elimiating each other's effects. Such issues are not addressed by this pass, but are subject to simplifications in
 * subsequent passes.
 */
struct RewriteToColumnarOpsPass : public PassWrapper<RewriteToColumnarOpsPass, OperationPass<ModuleOp>> {

    void runOnOperation() final;
};
} // namespace

void RewriteToColumnarOpsPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();

    // Rewrite elementwise comparisons, but only if the left-hand-side operand is a matrix with exactly one column and
    // the right-hand-side operand is a scalar.
    target.addDynamicallyLegalOp<daphne::EwGeOp, daphne::EwGtOp, daphne::EwLeOp, daphne::EwLtOp, daphne::EwEqOp,
                                 daphne::EwNeqOp>([](Operation *op) {
        Type lhsTy = op->getOperand(0).getType();
        Type rhsTy = op->getOperand(1).getType();
        return !(CompilerUtils::isMatTypeWithSingleCol(lhsTy) && CompilerUtils::isScaType(rhsTy));
    });
    // Rewrite elementwise logical operations, but only if their arguments are the results of columnar select ops or set
    // ops (these are the ops that result from relational algebra selection). However, depending on the order these
    // rewrite patterns are applied, the arguments of EwAndOp/EwOrOp may not have been rewritten yet. In that case, we
    // check if the arguments are defined by operations that would normally get rewritten by this pass.
    target.addDynamicallyLegalOp<daphne::EwAndOp, daphne::EwOrOp>([](Operation *op) {
        for (size_t i = 0; i < op->getNumOperands(); i++) {
            // Check each argument individually.
            Value arg = op->getOperand(i);
            if (auto plbmcOp = arg.getDefiningOp<daphne::ConvertPosListToBitmapOp>()) {
                // In case the inputs have already been rewritten.
                if (auto cOp = plbmcOp.getArg().getDefiningOp<daphne::CastOp>()) {
                    bool isArgColTy = cOp.getArg().getType().isa<daphne::ColumnType>();
                    bool isResMatTy = cOp.getRes().getType().isa<daphne::MatrixType>();
                    if (isArgColTy && isResMatTy) {
                        if (auto defOp = cOp.getArg().getDefiningOp()) {
                            // TODO We could define a trait for these ops.
                            if (!llvm::isa<daphne::ColSelectEqOp, daphne::ColSelectNeqOp, daphne::ColSelectGtOp,
                                           daphne::ColSelectGeOp, daphne::ColSelectLtOp, daphne::ColSelectLeOp,
                                           daphne::ColIntersectOp, daphne::ColMergeOp>(defOp))
                                continue;
                        }
                    }
                }
            } else if (auto defOp = arg.getDefiningOp()) {
                // In case the inputs have not been rewritten yet.
                // TODO Double-check if these ops are really illegal ones.
                if (llvm::isa<daphne::EwEqOp, daphne::EwNeqOp, daphne::EwGtOp, daphne::EwGeOp, daphne::EwLtOp,
                              daphne::EwLeOp, daphne::EwAndOp, daphne::EwOrOp>(defOp))
                    continue;
            }
            return true;
        }
        return false;
    });
    // Rewrite elementwise binary ops (other than comparisons), but only if both operands are a matrix with exactly one
    // column.
    target.addDynamicallyLegalOp<daphne::EwSubOp, daphne::EwMulOp>([](Operation *op) {
        Type lhsTy = op->getOperand(0).getType();
        Type rhsTy = op->getOperand(1).getType();
        return !(CompilerUtils::isMatTypeWithSingleCol(lhsTy) && CompilerUtils::isMatTypeWithSingleCol(rhsTy));
    });
    // Rewrite full aggregation ops, but only if the argument is a matrix with a single column.
    target.addDynamicallyLegalOp<daphne::AllAggSumOp>([](Operation *op) {
        Type argTy = op->getOperand(0).getType();
        return !CompilerUtils::isMatTypeWithSingleCol(argTy);
    });
    // Rewrite FilterRowOp and ExtractRowOp, but only if the source is a frame.
    // TODO We could also support matrix inputs.
    // TODO Check if the frame labels are known (current requirement of the rewrite code, which could be relaxed).
    target.addDynamicallyLegalOp<daphne::FilterRowOp, daphne::ExtractRowOp>([](Operation *op) {
        Type argTy = op->getOperand(0).getType();
        return !llvm::isa<daphne::FrameType>(argTy);
    });
    // Always rewrite InnerJoinOp and SemiJoinOp.
    target.addIllegalOp<daphne::InnerJoinOp, daphne::SemiJoinOp>();
    // Rewrite GroupOp, but only if all aggregation functions are SUM.
    // TODO We could also support other aggregation functions.
    target.addDynamicallyLegalOp<daphne::GroupOp>([](Operation *op) {
        auto gOp = llvm::dyn_cast<daphne::GroupOp>(op);
        ArrayAttr aggFuncs = gOp.getAggFuncs();
        return !llvm::all_of(aggFuncs.getValue(), [](Attribute af) {
            return af.dyn_cast<daphne::GroupEnumAttr>().getValue() == daphne::GroupEnum::SUM;
        });
        ;
    });

    patterns.add<ColumnarOpReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteToColumnarOpsPass() { return std::make_unique<RewriteToColumnarOpsPass>(); }
