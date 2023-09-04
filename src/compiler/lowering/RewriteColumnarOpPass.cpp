#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneInferFrameLabelsOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace mlir;

namespace
{
    void replaceIfCastOp(PatternRewriter &rewriter, mlir::daphne::CastOp &resultCast, mlir::Type resType, mlir::Operation *& op) {
        if(llvm::dyn_cast<mlir::daphne::CastOp>(op)) {
            resultCast = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(op, resType, op->getOperand(0));
        } else {
            resultCast = rewriter.create<mlir::daphne::CastOp>(op->getLoc(), resType, op->getResult(0));
        }
    }

    void extractAndProject(PatternRewriter &rewriter, std::string columnName, 
                        std::map<std::string, mlir::daphne::CastOp> &matrixColumns, 
                        std::map<std::string, mlir::daphne::ConstantOp> &labelColumns,
                        mlir::Type sourceType,
                        mlir::Type columnType,
                        mlir::Type finalType,
                        Operation * startOp,
                        mlir::Value sourceFrame,
                        mlir::OpResult positionList)
    {
        // create a constant op with the label of the column
        auto colName = rewriter.create<mlir::daphne::ConstantOp>(startOp->getLoc(), columnName);
        labelColumns.insert(std::pair<std::string,mlir::daphne::ConstantOp>{columnName, colName});
        // extract the column with the given label, cast it to column and project the position list onto it
        auto extract = rewriter.create<mlir::daphne::ExtractColOp>(startOp->getLoc(), sourceType, sourceFrame, colName);
        auto cast = rewriter.create<mlir::daphne::CastOp>(startOp->getLoc(), columnType, extract->getResult(0));
        auto project = rewriter.create<mlir::daphne::ColumnProjectOp>(startOp->getLoc(), columnType, cast->getResult(0), positionList);
        // cast the result back to a matrix and save result for later use
        mlir::daphne::CastOp castMatrix = rewriter.create<mlir::daphne::CastOp>(startOp->getLoc(), finalType, project->getResult(0));
        matrixColumns.insert(std::pair<std::string, mlir::daphne::CastOp>{columnName, castMatrix});
    }

    template <class DaphneCmp, class ColumnCmp>
    mlir::LogicalResult compareOp(PatternRewriter &rewriter, Operation *op) {
        DaphneCmp cmpOp = llvm::dyn_cast<DaphneCmp>(op);
        auto prevOp = op->getOperand(0).getDefiningOp();

        if(!cmpOp){
            return failure();
        }

        mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            rewriter.getContext(), vt
        );
        mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
            rewriter.getContext(), vt
        );

        mlir::daphne::CastOp cast;
        replaceIfCastOp(rewriter, cast, resType, prevOp);

        auto columnGe = rewriter.create<ColumnCmp>(prevOp->getLoc(), cast, cmpOp->getOperand(1));
        auto finalCast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), resTypeCast, columnGe->getResult(0));
        auto numRows = rewriter.create<mlir::daphne::NumRowsOp>(prevOp->getLoc(), rewriter.getIndexType(), cast->getOperand(0));
        auto res = rewriter.replaceOpWithNewOp<mlir::daphne::PositionListBitmapConverterOp>(cmpOp, resTypeCast, finalCast->getResult(0), numRows);
        cmpOp->getResult(0).replaceAllUsesWith(res->getResult(0));
        return success();
    }

    template <class DaphneBinaryOp, class ColumnBinaryOp>
    mlir::LogicalResult binaryOp(PatternRewriter &rewriter, Operation *op) {
        DaphneBinaryOp binaryOp = llvm::dyn_cast<DaphneBinaryOp>(op);
        auto prevOpLhs = op->getOperand(0).getDefiningOp();

        auto prevOpRhs = op->getOperand(1).getDefiningOp();

        if(!binaryOp){
            return failure();
        }

        mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            rewriter.getContext(), vt
        );
        mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
            rewriter.getContext(), vt
        );

        mlir::daphne::CastOp castLhs;
        replaceIfCastOp(rewriter, castLhs, resType, prevOpLhs);
        mlir::daphne::CastOp castRhs;
        replaceIfCastOp(rewriter, castRhs, resType, prevOpRhs);

        auto colBinaryOp = rewriter.create<ColumnBinaryOp>(binaryOp->getLoc(), resType, castLhs, castRhs);
        auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(binaryOp, resTypeCast, colBinaryOp->getResult(0));
        binaryOp->getResult(0).replaceAllUsesWith(res->getResult(0));
        return success();
    }

    template <class DaphneAgg, class ColumnAgg>
    mlir::LogicalResult allAggOp(PatternRewriter &rewriter, Operation *op) {
        DaphneAgg aggOp = llvm::dyn_cast<DaphneAgg>(op);
        auto prevOp = op->getOperand(0).getDefiningOp();

        if(!aggOp){
            return failure();
        }

        mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            rewriter.getContext(), vt
        );
        mlir::Type resTypeCast = mlir::daphne::MatrixType::get(
            rewriter.getContext(), vt
        );

        mlir::daphne::CastOp cast;
        replaceIfCastOp(rewriter, cast, resType, prevOp);

        mlir::Type input = cast->getResult(0).getType().dyn_cast<mlir::daphne::ColumnType>().getColumnType();
        auto columnAgg = rewriter.replaceOpWithNewOp<ColumnAgg>(aggOp, input, cast->getResult(0));
        auto uses = aggOp->getUses();
        for (auto x = uses.begin(); x != uses.end(); x++) {
            x->getOwner()->replaceUsesOfWith(aggOp, columnAgg);
            mlir::daphne::CastOp removeCast = dyn_cast<mlir::daphne::CastOp>(x->getOwner());
            if(!removeCast) {
                return failure();
            }
            auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(removeCast, resTypeCast, columnAgg->getResult(0));
            removeCast->getResult(0).replaceAllUsesWith(res->getResult(0));
        }
        return success();
    }

    template <class SourceOp>
    mlir::LogicalResult createFramesForSuccessors(
        PatternRewriter &rewriter, 
        SourceOp sourceOp, 
        std::map<Operation*, std::set<std::string>> columnNamesPerSuccessor, 
        std::set<std::string> usedColumnNames, 
        std::map<std::string, mlir::daphne::CastOp> matrixColumns, 
        std::map<std::string, mlir::daphne::ConstantOp> labelColumns
    ) {
        // Create a frame with all needed columns for each successor of the source Op
        for (auto it = columnNamesPerSuccessor.begin(); it != columnNamesPerSuccessor.end(); it++) {
            std::set<std::string> columnNames;
            // check if columns available in input frame
            for (auto columnName : it->second) {
                if (std::find(usedColumnNames.begin(), usedColumnNames.end(), columnName) != usedColumnNames.end()) {
                    columnNames.insert(columnName);
                }
            }

            auto successorOperation = it->first;

            std::vector<mlir::Type> colTypes;
            std::vector<mlir::Value> cols;
            std::vector<mlir::Value> labels;
            std::vector<std::string> *labelStrings = new std::vector<std::string>();

            // Get the needed input matrix, labels and types for the frame
            for (size_t i = 0; i < columnNames.size(); i++) {
                mlir::daphne::CastOp castMatrix = matrixColumns.find(*std::next(columnNames.begin(), i))->second;
                mlir::daphne::ConstantOp columnNameOp = labelColumns.find(*std::next(columnNames.begin(), i))->second;
                cols.push_back(castMatrix->getResult(0));
                labels.push_back(columnNameOp->getResult(0));
                colTypes.push_back(castMatrix->getResult(0).getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());
                labelStrings->push_back(columnNameOp->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            }
            mlir::Type resTypeFrame = mlir::daphne::FrameType::get(
            rewriter.getContext(), colTypes
            ).withLabels(labelStrings);

            mlir::daphne::CreateFrameOp res;
            // We need to replace the innerJoinOp with the CreateFrameOp of the final successor to create a valid IR
            if(it == std::prev(columnNamesPerSuccessor.end())) {
                res = rewriter.replaceOpWithNewOp<mlir::daphne::CreateFrameOp>(sourceOp, resTypeFrame, cols, labels);
            } else {
                res = rewriter.create<mlir::daphne::CreateFrameOp>(sourceOp->getLoc(), resTypeFrame, cols, labels);
            }

            successorOperation->replaceUsesOfWith(sourceOp, res);
        }
        return success();
    }

    bool checkBitmapOutputOp(Operation *op) {
        if(dyn_cast<mlir::daphne::EwEqOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwNeqOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwGtOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwGeOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwLtOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwLeOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwAndOp>(op)) {
            return true;
        } else if(dyn_cast<mlir::daphne::EwOrOp>(op)) {
            return true;
        } else {
            return false;
        }
    }

    void dfsSuccessors(
        Operation *op,
        Operation *parent,
        std::set<Operation*> &visited,
        std::map<Operation*, std::set<std::string>> &columnNamesPerSuccessor,
        std::set<std::string> &distinctColumnNames
    ) {
        if(visited.find(op) != visited.end()) {
            return;
        }

        // We don't want to visit the successors of bitmap output ops to make sure,
        // that the following names are added to the correct datapath
        if(checkBitmapOutputOp(op)) {
            return;
        }

        visited.insert(op);
        if(llvm::dyn_cast<mlir::daphne::ExtractColOp>(op)) {
            Operation * constantOp = op->getOperand(1).getDefiningOp();
            columnNamesPerSuccessor[parent].insert(constantOp->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            distinctColumnNames.insert(constantOp->getAttr("value").cast<mlir::StringAttr>().getValue().str());
        }else if(llvm::dyn_cast<mlir::daphne::PrintOp>(op)) {
            auto colNamesPrint = op->getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>().getLabels();
            for (auto label : *colNamesPrint) {
                columnNamesPerSuccessor[parent].insert(label);
                distinctColumnNames.insert(label);
            }
        } else if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op)) {
            columnNamesPerSuccessor[parent].insert(op->getOperand(2).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            distinctColumnNames.insert(op->getOperand(2).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            columnNamesPerSuccessor[parent].insert(op->getOperand(3).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            distinctColumnNames.insert(op->getOperand(3).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str());
        }

        if (llvm::dyn_cast<mlir::daphne::ExtractColOp>(op)) {
            return;
        }

        for (auto indexedResult : llvm::enumerate(op->getResults())) {
            Value result = indexedResult.value();
            for (Operation *userOp : result.getUsers()) {
                dfsSuccessors(userOp, parent, visited, columnNamesPerSuccessor, distinctColumnNames);
            }
        }
    }

    struct ColumnarOpReplacement : public RewritePattern{

        ColumnarOpReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if(llvm::dyn_cast<mlir::daphne::EwGeOp>(op)){
                return compareOp<mlir::daphne::EwGeOp, mlir::daphne::ColumnGeOp>(rewriter, op);
            } else if(llvm::dyn_cast<mlir::daphne::EwGtOp>(op)) {
                return compareOp<mlir::daphne::EwGtOp, mlir::daphne::ColumnGtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLeOp>(op)) {
                return compareOp<mlir::daphne::EwLeOp, mlir::daphne::ColumnLeOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLtOp>(op)) {
                return compareOp<mlir::daphne::EwLtOp, mlir::daphne::ColumnLtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwEqOp>(op)) {
                return compareOp<mlir::daphne::EwEqOp, mlir::daphne::ColumnEqOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwNeqOp>(op)) {
                return compareOp<mlir::daphne::EwNeqOp, mlir::daphne::ColumnNeqOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::FilterRowOp>(op)) {
                mlir::daphne::FilterRowOp filterOp = llvm::dyn_cast<mlir::daphne::FilterRowOp>(op);
                if(!filterOp){
                    return failure();
                };
                auto x = filterOp.getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>();
                auto sourceColumnLabels = x.getLabels();

                std::map<Operation*, std::set<std::string>> columnNamesPerSuccessor;
                std::set<std::string> distinctColumnNames;
                std::set<Operation*> visited;

                for (auto indexedResult : llvm::enumerate(filterOp->getResults())) {
                    Value result = indexedResult.value();
                    visited.insert(result.getDefiningOp());
                    for (Operation *userOp : result.getUsers()) {
                        columnNamesPerSuccessor.insert({userOp ,std::set<std::string>() });
                        dfsSuccessors(userOp, userOp, visited, columnNamesPerSuccessor, distinctColumnNames);
                    }
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeMatrix = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeFrame = mlir::daphne::FrameType::get(
                        rewriter.getContext(), {vt}
                );

                auto posList = rewriter.create<mlir::daphne::BitmapPositionListConverterOp>(filterOp->getLoc(), resTypeMatrix, filterOp->getOperand(1));
                auto pos = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resType, posList->getResult(0));

                std::map<std::string, mlir::daphne::CastOp> matrixColumns;
                std::map<std::string, mlir::daphne::ConstantOp> labelColumns;
                std::set<std::string> usedColumnNames;

                for (std::string columnName : distinctColumnNames) {
                    // check if columns available in input frame
                    if (std::find(sourceColumnLabels->begin(), sourceColumnLabels->end(), columnName) == sourceColumnLabels->end()) {
                        continue;
                    }
                    usedColumnNames.insert(columnName);
                    extractAndProject(rewriter, columnName, matrixColumns, labelColumns, resTypeFrame,
                     resType, resTypeMatrix, filterOp, filterOp->getOperand(0), pos->getResult(0));
                }

                return createFramesForSuccessors<mlir::daphne::FilterRowOp>(rewriter, filterOp, columnNamesPerSuccessor, usedColumnNames, matrixColumns, labelColumns);
            }else if(llvm::dyn_cast<mlir::daphne::EwMulOp>(op)) {
                return binaryOp<mlir::daphne::EwMulOp, mlir::daphne::ColumnMulOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwAndOp>(op)) {
                return binaryOp<mlir::daphne::EwAndOp, mlir::daphne::ColumnAndOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::AllAggSumOp>(op)) {
                return allAggOp<mlir::daphne::AllAggSumOp, mlir::daphne::ColumnAggSumOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op)) {
                mlir::daphne::InnerJoinOp joinOp = llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op);
                if(!joinOp){
                    return failure();
                };
                auto lhs = joinOp.getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>();
                auto lhs_labels = lhs.getLabels();
                auto rhs = joinOp.getOperand(1).getType().dyn_cast<mlir::daphne::FrameType>();
                auto rhs_labels = rhs.getLabels();
                auto lhs_col_label_op = joinOp.getOperand(2).getDefiningOp();
                auto rhs_col_label_op = joinOp.getOperand(3).getDefiningOp();
                auto lhs_col_label = joinOp.getOperand(2).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();
                auto rhs_col_label = joinOp.getOperand(3).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();

                std::map<Operation*, std::set<std::string>> columnNamesPerSuccessor;
                std::set<std::string> distinctColumnNames;
                std::set<Operation*> visited;

                for (auto indexedResult : llvm::enumerate(joinOp->getResults())) {
                    Value result = indexedResult.value();
                    visited.insert(result.getDefiningOp());
                    for (Operation *userOp : result.getUsers()) {
                        columnNamesPerSuccessor.insert({userOp ,std::set<std::string>() });
                        dfsSuccessors(userOp, userOp, visited, columnNamesPerSuccessor, distinctColumnNames);
                    }
                }

                mlir::Type vt = mlir::daphne::UnknownType::get(rewriter.getContext());
                mlir::Type resType = mlir::daphne::ColumnType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeMatrix = mlir::daphne::MatrixType::get(
                    rewriter.getContext(), vt
                );
                mlir::Type resTypeFrame = mlir::daphne::FrameType::get(
                        rewriter.getContext(), {vt}
                );

                std::vector<mlir::Type> joinResultType{resType, resType};

                auto lhs_single_frame = rewriter.create<mlir::daphne::ExtractColOp>(joinOp->getLoc(), resTypeFrame, joinOp.getOperand(0), lhs_col_label_op->getResult(0));
                auto rhs_single_frame = rewriter.create<mlir::daphne::ExtractColOp>(joinOp->getLoc(), resTypeFrame, joinOp.getOperand(1), rhs_col_label_op->getResult(0));
                auto lhs_column = rewriter.create<mlir::daphne::CastOp>(joinOp->getLoc(), resType, lhs_single_frame->getResult(0));
                auto rhs_column = rewriter.create<mlir::daphne::CastOp>(joinOp->getLoc(), resType, rhs_single_frame->getResult(0));
                auto join = rewriter.create<mlir::daphne::ColumnJoinOp>(joinOp->getLoc(), joinResultType, lhs_column->getResult(0), rhs_column->getResult(0));

                std::map<std::string, mlir::daphne::CastOp> matrixColumns;
                std::map<std::string, mlir::daphne::ConstantOp> labelColumns;
                std::set<std::string> usedColumnNames;
                for (std::string columnName : distinctColumnNames) {
                    // check if columns available in input frame
                    if (std::find(lhs_labels->begin(), lhs_labels->end(), columnName) == lhs_labels->end()
                        && std::find(rhs_labels->begin(), rhs_labels->end(), columnName) == rhs_labels->end()) {
                        continue;
                    } else if (std::find(lhs_labels->begin(), lhs_labels->end(), columnName) != lhs_labels->end()) {
                        usedColumnNames.insert(columnName);
                        extractAndProject(rewriter, columnName, matrixColumns, labelColumns, resTypeFrame,
                         resType, resTypeMatrix, joinOp, joinOp.getOperand(0), join->getResult(0));
                    } else {
                        usedColumnNames.insert(columnName);
                        extractAndProject(rewriter, columnName, matrixColumns, labelColumns, resTypeFrame,
                         resType, resTypeMatrix, joinOp, joinOp.getOperand(1), join->getResult(1));
                    }
                    

                }

                return createFramesForSuccessors<mlir::daphne::InnerJoinOp>(rewriter, joinOp, columnNamesPerSuccessor, usedColumnNames, matrixColumns, labelColumns);
            }
        }
    };

    struct RewriteColumnarOpPass : public PassWrapper<RewriteColumnarOpPass, OperationPass<ModuleOp>> {
    
    void runOnOperation() final;
    
    };
}

void RewriteColumnarOpPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addIllegalOp<mlir::daphne::EwGeOp, mlir::daphne::EwGtOp, mlir::daphne::EwLeOp, mlir::daphne::EwLtOp, mlir::daphne::EwEqOp, mlir::daphne::EwNeqOp, mlir::daphne::FilterRowOp,
                        mlir::daphne::EwMulOp, mlir::daphne::EwAndOp, mlir::daphne::AllAggSumOp, mlir::daphne::InnerJoinOp>();

    patterns.add<ColumnarOpReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteColumnarOpPass()
{
    return std::make_unique<RewriteColumnarOpPass>();
}

