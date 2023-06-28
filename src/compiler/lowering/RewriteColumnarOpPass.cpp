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
    template <class DaphneCmp, class ColumnCmp>
    mlir::LogicalResult compareOp(PatternRewriter &rewriter, Operation *op) {
        DaphneCmp cmpOp = llvm::dyn_cast<DaphneCmp>(op);
        mlir::Value cmpInout = op->getOperand(0);
        auto prevOp = cmpInout.getDefiningOp();

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
        if(llvm::dyn_cast<mlir::daphne::CastOp>(prevOp)) {
            cast = rewriter.replaceOpWithNewOp<mlir::daphne::CastOp>(prevOp, resType, prevOp->getOperand(0));
        } else {
            cast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), resType, prevOp->getResult(0));
        }
        auto columnGe = rewriter.create<ColumnCmp>(prevOp->getLoc(), cast, cmpOp->getOperand(1));
        auto finalCast = rewriter.create<mlir::daphne::CastOp>(prevOp->getLoc(), resTypeCast, columnGe->getResult(0));
        auto numRows = rewriter.create<mlir::daphne::NumRowsOp>(prevOp->getLoc(), rewriter.getIndexType(), cast->getOperand(0));
        auto res = rewriter.replaceOpWithNewOp<mlir::daphne::PositionListBitmapConverterOp>(cmpOp, resTypeCast, finalCast->getResult(0), numRows);
        auto uses = cmpOp->getUses();
        for (auto x = uses.begin(); x != uses.end(); x++) {
            x->getOwner()->replaceUsesOfWith(cmpOp, res);
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
            } else if(llvm::dyn_cast<mlir::daphne::FilterRowOp>(op)) {
                mlir::daphne::FilterRowOp filterOp = llvm::dyn_cast<mlir::daphne::FilterRowOp>(op);
                if(!filterOp){
                    return failure();
                };
                auto x = filterOp.getOperand(0).getType().dyn_cast<mlir::daphne::FrameType>();
                auto y = x.getLabels();
                for (auto label : *y) {
                    std::cout << label << std::endl;
                }
                std::cout << filterOp->getNumResults() << std::endl;
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

                for (auto it = columnNamesPerSuccessor.begin(); it != columnNamesPerSuccessor.end(); it++) {
                     llvm::outs() << "Successor: " << it->first->getName() << "\n";
                    for (auto columnName : it->second) {
                        std::cout << columnName << std::endl;
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

                for (std::string columnName : distinctColumnNames) {
                    // check if columns available in input frame
                    if (std::find(y->begin(), y->end(), columnName) == y->end()) {
                        continue;
                    }
                    // create a constant op with the label of the column
                    auto colName = rewriter.create<mlir::daphne::ConstantOp>(filterOp->getLoc(), columnName);
                    labelColumns.insert(std::pair<std::string,mlir::daphne::ConstantOp>{columnName, colName});
                    // extract the column with the given label, cast it to column and project the position list onto it
                    auto extract = rewriter.create<mlir::daphne::ExtractColOp>(filterOp->getLoc(), resTypeFrame, filterOp->getOperand(0), colName);
                    auto cast = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resType, extract->getResult(0));
                    auto project = rewriter.create<mlir::daphne::ColumnProjectOp>(filterOp->getLoc(), resType, cast->getResult(0), pos->getResult(0));
                    // cast the result back to a matrix and save result for later use
                    mlir::daphne::CastOp castMatrix = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resTypeMatrix, project->getResult(0));
                    matrixColumns.insert(std::pair<std::string, mlir::daphne::CastOp>{columnName, castMatrix});

                }

                // Create a frame with all needed columns for each successor of the filterRowOp
                for (auto it = columnNamesPerSuccessor.begin(); it != columnNamesPerSuccessor.end(); it++) {
                    std::set<std::string> columnNames;
                    // check if columns available in input frame
                    for (auto columnName : it->second) {
                        if (std::find(y->begin(), y->end(), columnName) != y->end()) {
                            columnNames.insert(columnName);
                            std::cout <<"name " << columnName << std::endl;
                        }
                    }

                    auto successorOperation = it->first;

                    std::vector<mlir::Type> colTypes;
                    std::vector<mlir::Value> cols;
                    std::vector<mlir::Value> labels;

                    // Get the needed input matrix, labels and types for the frame
                    for (size_t i = 0; i < columnNames.size(); i++) {
                        mlir::daphne::CastOp castMatrix = matrixColumns.find(*std::next(columnNames.begin(), i))->second;
                        mlir::daphne::ConstantOp columnNameOp = labelColumns.find(*std::next(columnNames.begin(), i))->second;
                        cols.push_back(castMatrix->getResult(0));
                        labels.push_back(columnNameOp->getResult(0));
                        colTypes.push_back(castMatrix->getResult(0).getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());
                    }
                    mlir::Type resTypeFrame = mlir::daphne::FrameType::get(
                    rewriter.getContext(), colTypes
                    );

                    mlir::daphne::CreateFrameOp res;
                    // We need to replace the filterRowOp with the CreateFrameOp of the final successor to create a valid IR
                    if(it == std::prev(columnNamesPerSuccessor.end())) {
                        res = rewriter.replaceOpWithNewOp<mlir::daphne::CreateFrameOp>(filterOp, resTypeFrame, cols, labels);
                    } else {
                        res = rewriter.create<mlir::daphne::CreateFrameOp>(filterOp->getLoc(), resTypeFrame, cols, labels);
                    }

                    successorOperation->replaceUsesOfWith(filterOp, res);
                
                }

                return success();
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
    target.addIllegalOp<mlir::daphne::EwGeOp, mlir::daphne::EwGtOp, mlir::daphne::EwLeOp, mlir::daphne::EwLtOp, mlir::daphne::EwEqOp, mlir::daphne::EwNeqOp, mlir::daphne::FilterRowOp>();

    patterns.add<ColumnarOpReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteColumnarOpPass()
{
    return std::make_unique<RewriteColumnarOpPass>();
}

