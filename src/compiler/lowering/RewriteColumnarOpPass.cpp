#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneInferFrameLabelsOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
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
    //helper functions
    mlir::Type getUnknownType(PatternRewriter &rewriter) {
        return mlir::daphne::UnknownType::get(rewriter.getContext());
    }

    mlir::Type getColumnType(PatternRewriter &rewriter) {
        return mlir::daphne::ColumnType::get(
            rewriter.getContext(), getUnknownType(rewriter)
        );
    }

    mlir::Type getMatrixType(PatternRewriter &rewriter) {
        return mlir::daphne::MatrixType::get(
            rewriter.getContext(), getUnknownType(rewriter)
        );
    }

    mlir::Type getFrameType(PatternRewriter &rewriter) {
        return mlir::daphne::FrameType::get(
            rewriter.getContext(), {getUnknownType(rewriter)}
        );
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

    //replacement functions

    void replaceIfCastOp(
        PatternRewriter &rewriter,
        mlir::daphne::CastOp &resultCast, 
        mlir::Type resType, 
        mlir::Operation *& op
    ) {
        if(llvm::dyn_cast<mlir::daphne::CastOp>(op)) {
            resultCast = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(op, resType, op->getOperand(0));
        } else {
            resultCast = rewriter.create<mlir::daphne::CastOp>(op->getLoc(), resType, op->getResult(0));
        }
    }

    // Extract all columns from a frame, cast them to columns and project the position list onto them and convert result back to a matrix
    void extractAndProject(
        PatternRewriter &rewriter,
        std::string columnName, 
        std::map<std::string, mlir::daphne::CastOp> &matrixColumns, 
        std::map<std::string, mlir::daphne::ConstantOp> &labelColumns,
        mlir::Type sourceType,
        mlir::Type columnType,
        mlir::Type finalType,
        Operation * startOp,
        mlir::Value sourceFrame,
        mlir::OpResult positionList
    ) {
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

    // Replace comparisons by casting the incoming data to a column, converting the result back to a matrix and converting the resulting
    // position list to a bitmap
    template <class DaphneCmp, class ColumnCmp>
    mlir::LogicalResult replaceCompareOp(
        PatternRewriter &rewriter, 
        Operation *op
    ) {
        DaphneCmp cmpOp = llvm::dyn_cast<DaphneCmp>(op);
        auto prevOp = op->getOperand(0).getDefiningOp();

        if(!cmpOp){
            return failure();
        }

        mlir::daphne::CastOp cast;
        replaceIfCastOp(rewriter, cast, getColumnType(rewriter), prevOp);

        auto columnGe = rewriter.create<ColumnCmp>(prevOp->getLoc(), cast, cmpOp->getOperand(1));
        auto finalCast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), getMatrixType(rewriter), columnGe->getResult(0));
        auto numRows = rewriter.create<mlir::daphne::NumRowsOp>(prevOp->getLoc(), rewriter.getIndexType(), cast->getOperand(0));
        auto res = rewriter.replaceOpWithNewOp<mlir::daphne::PositionListBitmapConverterOp>(cmpOp, getMatrixType(rewriter), finalCast->getResult(0), numRows);
        cmpOp->getResult(0).replaceAllUsesWith(res->getResult(0));
        return success();
    }

    // Replace comparisons by casting the incoming data to columns and converting the result back to a matrix
    template <class DaphneBinaryOp, class ColumnBinaryOp>
    mlir::LogicalResult replaceBinaryOp(
        PatternRewriter &rewriter,
        Operation *op
    ) {
        DaphneBinaryOp binaryOp = llvm::dyn_cast<DaphneBinaryOp>(op);
        auto prevOpLhs = op->getOperand(0).getDefiningOp();

        auto prevOpRhs = op->getOperand(1).getDefiningOp();

        if(!binaryOp){
            return failure();
        }

        mlir::daphne::CastOp castLhs;
        replaceIfCastOp(rewriter, castLhs, getColumnType(rewriter), prevOpLhs);
        mlir::daphne::CastOp castRhs;
        replaceIfCastOp(rewriter, castRhs, getColumnType(rewriter), prevOpRhs);

        auto colBinaryOp = rewriter.create<ColumnBinaryOp>(binaryOp->getLoc(), getColumnType(rewriter), castLhs, castRhs);
        auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(binaryOp, getMatrixType(rewriter), colBinaryOp->getResult(0));
        binaryOp->getResult(0).replaceAllUsesWith(res->getResult(0));
        return success();
    }

    // Replace aggregations by casting the incoming data to a column, converting the result back to a matrix and removing
    // unnecessary casts which where necessary due to the result of DaphneAgg being a scalar value
    template <class DaphneAgg, class ColumnAgg>
    mlir::LogicalResult replaceAllAggOp(
        PatternRewriter &rewriter, 
        Operation *op
    ) {
        DaphneAgg aggOp = llvm::dyn_cast<DaphneAgg>(op);
        auto prevOp = op->getOperand(0).getDefiningOp();

        if(!aggOp){
            return failure();
        }

        mlir::daphne::CastOp cast;
        replaceIfCastOp(rewriter, cast, getColumnType(rewriter), prevOp);

        mlir::Type input = cast->getResult(0).getType().dyn_cast<mlir::daphne::ColumnType>().getColumnType();
        auto columnAgg = rewriter.replaceOpWithNewOp<ColumnAgg>(aggOp, input, cast->getResult(0));
        auto uses = aggOp->getUses();
        for (auto x = uses.begin(); x != uses.end(); x++) {
            x->getOwner()->replaceUsesOfWith(aggOp, columnAgg);
            mlir::daphne::CastOp removeCast = dyn_cast<mlir::daphne::CastOp>(x->getOwner());
            if(!removeCast) {
                return failure();
            }
            auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(removeCast, getMatrixType(rewriter), columnAgg->getResult(0));
            removeCast->getResult(0).replaceAllUsesWith(res->getResult(0));
        }
        return success();
    }

    // Combine projected matrixes to frames with the needed columns calculated in the DFS for each successor of the source Op
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
        
        // Get the column names of the current operation if it has one
        if(llvm::dyn_cast<mlir::daphne::ExtractColOp>(op)) {
            auto constantOpName = op->getOperand(1).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();
            columnNamesPerSuccessor[parent].insert(constantOpName);
            distinctColumnNames.insert(constantOpName);
        }else if(llvm::dyn_cast<mlir::daphne::PrintOp>(op)) {
            auto colNamesPrint = op->getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>().getLabels();
            for (auto label : *colNamesPrint) {
                columnNamesPerSuccessor[parent].insert(label);
                distinctColumnNames.insert(label);
            }
        } else if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op)) {
            auto joinColumnName1 = op->getOperand(2).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();
            columnNamesPerSuccessor[parent].insert(joinColumnName1);
            distinctColumnNames.insert(joinColumnName1);
            auto joinColumnName2 = op->getOperand(3).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();
            columnNamesPerSuccessor[parent].insert(joinColumnName2);
            distinctColumnNames.insert(joinColumnName2);
        }

        // We don't want to visit the successors of ExtractColOps as they will
        // only contain this one specific column or get other ones from different parent ops
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

    // Start the DFS for each result of the sourceOp
    void startDfsSuccessors(
        std::map<Operation*, std::set<std::string>> &columnNamesPerSuccessor,
        std::set<std::string> &distinctColumnNames,
        std::set<Operation*> &visited,
        Operation *sourceOp
    ) {
        for (auto indexedResult : llvm::enumerate(sourceOp->getResults())) {
            Value result = indexedResult.value();
            visited.insert(result.getDefiningOp());
            for (Operation *userOp : result.getUsers()) {
                columnNamesPerSuccessor.insert({userOp ,std::set<std::string>() });
                dfsSuccessors(userOp, userOp, visited, columnNamesPerSuccessor, distinctColumnNames);
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
                return replaceCompareOp<mlir::daphne::EwGeOp, mlir::daphne::ColumnGeOp>(rewriter, op);
            } else if(llvm::dyn_cast<mlir::daphne::EwGtOp>(op)) {
                return replaceCompareOp<mlir::daphne::EwGtOp, mlir::daphne::ColumnGtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLeOp>(op)) {
                return replaceCompareOp<mlir::daphne::EwLeOp, mlir::daphne::ColumnLeOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwLtOp>(op)) {
                return replaceCompareOp<mlir::daphne::EwLtOp, mlir::daphne::ColumnLtOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwEqOp>(op)) {
                return replaceCompareOp<mlir::daphne::EwEqOp, mlir::daphne::ColumnEqOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwNeqOp>(op)) {
                return replaceCompareOp<mlir::daphne::EwNeqOp, mlir::daphne::ColumnNeqOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::FilterRowOp>(op)) {
                mlir::daphne::FilterRowOp filterOp = llvm::dyn_cast<mlir::daphne::FilterRowOp>(op);
                if(!filterOp){
                    return failure();
                };

                auto inputFrameType = filterOp.getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>();
                auto sourceColumnLabels = inputFrameType.getLabels();

                std::map<Operation*, std::set<std::string>> columnNamesPerSuccessor;
                std::set<std::string> distinctColumnNames;
                std::set<Operation*> visited;

                startDfsSuccessors(columnNamesPerSuccessor, distinctColumnNames, visited, filterOp);

                // Convert the incoming bitmap to a position list
                auto posList = rewriter.create<mlir::daphne::BitmapPositionListConverterOp>(filterOp->getLoc(), getMatrixType(rewriter), filterOp->getOperand(1));
                auto pos = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), getColumnType(rewriter), posList->getResult(0));

                // Run the necessary casts and projections for each column needed by the successors
                std::map<std::string, mlir::daphne::CastOp> matrixColumns;
                std::map<std::string, mlir::daphne::ConstantOp> labelColumns;
                std::set<std::string> usedColumnNames;
                for (std::string columnName : distinctColumnNames) {
                    // check if columns available in input frame
                    if (std::find(sourceColumnLabels->begin(), sourceColumnLabels->end(), columnName) == sourceColumnLabels->end()) {
                        continue;
                    }
                    usedColumnNames.insert(columnName);
                    extractAndProject(rewriter, columnName, matrixColumns, labelColumns, getFrameType(rewriter),
                     getColumnType(rewriter), getMatrixType(rewriter), 
                     filterOp, filterOp->getOperand(0), pos->getResult(0));
                }

                return createFramesForSuccessors<mlir::daphne::FilterRowOp>(rewriter, filterOp, columnNamesPerSuccessor, usedColumnNames, 
                                                                            matrixColumns, labelColumns);
            }else if(llvm::dyn_cast<mlir::daphne::EwMulOp>(op)) {
                return replaceBinaryOp<mlir::daphne::EwMulOp, mlir::daphne::ColumnMulOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::EwAndOp>(op)) {
                return replaceBinaryOp<mlir::daphne::EwAndOp, mlir::daphne::ColumnAndOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::AllAggSumOp>(op)) {
                return replaceAllAggOp<mlir::daphne::AllAggSumOp, mlir::daphne::ColumnAggSumOp>(rewriter, op);
            }else if(llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op)) {
                mlir::daphne::InnerJoinOp joinOp = llvm::dyn_cast<mlir::daphne::InnerJoinOp>(op);
                if(!joinOp){
                    return failure();
                };
                auto lhs = joinOp.getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>();
                auto lhsLabels = lhs.getLabels();
                auto rhs = joinOp.getOperand(1).getType().dyn_cast<mlir::daphne::FrameType>();
                auto rhsLabels = rhs.getLabels();
                auto lhsColLabelOp = joinOp.getOperand(2).getDefiningOp();
                auto rhsColLabelOp = joinOp.getOperand(3).getDefiningOp();
                auto lhsColLabel = joinOp.getOperand(2).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();
                auto rhsColLabel = joinOp.getOperand(3).getDefiningOp()->getAttr("value").cast<mlir::StringAttr>().getValue().str();

                std::map<Operation*, std::set<std::string>> columnNamesPerSuccessor;
                std::set<std::string> distinctColumnNames;
                std::set<Operation*> visited;

                startDfsSuccessors(columnNamesPerSuccessor, distinctColumnNames, visited, joinOp);

                // Run the actual join
                std::vector<mlir::Type> joinResultType{getColumnType(rewriter), getColumnType(rewriter)};

                auto lhsSingleFrame = rewriter.create<mlir::daphne::ExtractColOp>(joinOp->getLoc(), getFrameType(rewriter), 
                                                    joinOp.getOperand(0), lhsColLabelOp->getResult(0));
                auto rhsSingleFrame = rewriter.create<mlir::daphne::ExtractColOp>(joinOp->getLoc(), getFrameType(rewriter), 
                                                    joinOp.getOperand(1), rhsColLabelOp->getResult(0));
                auto lhsColumn = rewriter.create<mlir::daphne::CastOp>(joinOp->getLoc(), getColumnType(rewriter), lhsSingleFrame->getResult(0));
                auto rhsColumn = rewriter.create<mlir::daphne::CastOp>(joinOp->getLoc(), getColumnType(rewriter), rhsSingleFrame->getResult(0));
                auto join = rewriter.create<mlir::daphne::ColumnJoinOp>(joinOp->getLoc(), joinResultType, lhsColumn->getResult(0), rhsColumn->getResult(0));

                // Run the necessary casts and projections for each column needed by the successors
                std::map<std::string, mlir::daphne::CastOp> matrixColumns;
                std::map<std::string, mlir::daphne::ConstantOp> labelColumns;
                std::set<std::string> usedColumnNames;
                for (std::string columnName : distinctColumnNames) {
                    // check if columns available in input frame
                    if (std::find(lhsLabels->begin(), lhsLabels->end(), columnName) == lhsLabels->end()
                        && std::find(rhsLabels->begin(), rhsLabels->end(), columnName) == rhsLabels->end()) {
                        continue;
                    } else if (std::find(lhsLabels->begin(), lhsLabels->end(), columnName) != lhsLabels->end()) {
                        usedColumnNames.insert(columnName);
                        extractAndProject(rewriter, columnName, matrixColumns, labelColumns, getFrameType(rewriter),
                         getColumnType(rewriter), getMatrixType(rewriter), joinOp, joinOp.getOperand(0), join->getResult(0));
                    } else {
                        usedColumnNames.insert(columnName);
                        extractAndProject(rewriter, columnName, matrixColumns, labelColumns, getFrameType(rewriter),
                         getColumnType(rewriter), getMatrixType(rewriter), joinOp, joinOp.getOperand(1), join->getResult(1));
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

