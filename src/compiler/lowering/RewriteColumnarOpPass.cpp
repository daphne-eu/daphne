#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

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
        std::map<Operation*, std::vector<std::string>> &columnNamesPerSuccessor
    ) {
        if(visited.find(op) != visited.end()) {
            return;
        }
        visited.insert(op);
        for (auto indexedResult : llvm::enumerate(op->getResults())) {
            Value result = indexedResult.value();
            if(llvm::dyn_cast<mlir::daphne::ExtractColOp>(op)) {
                Operation * constantOp = op->getOperand(1).getDefiningOp();
                columnNamesPerSuccessor[parent].push_back(constantOp->getAttr("value").cast<mlir::StringAttr>().getValue().str());
            }
            for (Operation *userOp : result.getUsers()) {
                dfsSuccessors(userOp, parent, visited, columnNamesPerSuccessor);
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
                std::vector<std::string> columnNames;
                std::cout << filterOp->getNumResults() << std::endl;
                Operation * currentOp = filterOp;
                std::map<Operation*, std::vector<std::string>> columnNamesPerSuccessor;
                std::set<Operation*> visited;
                while (currentOp->getNumResults()) {
                    for (auto indexedResult : llvm::enumerate(currentOp->getResults())) {
                        Value result = indexedResult.value();
                        for (Operation *userOp : result.getUsers()) {
                            if(llvm::dyn_cast<mlir::daphne::ExtractColOp>(userOp)) {
                                Operation * constantOp = userOp->getOperand(1).getDefiningOp();
                                columnNames.push_back(constantOp->getAttr("value").cast<mlir::StringAttr>().getValue().str());
                            }
                            currentOp = userOp;
                        }
                    }
                }

                for (auto indexedResult : llvm::enumerate(filterOp->getResults())) {
                    Value result = indexedResult.value();
                    visited.insert(result.getDefiningOp());
                    for (Operation *userOp : result.getUsers()) {
                        columnNamesPerSuccessor.insert({userOp ,std::vector<std::string>() });
                        dfsSuccessors(userOp, userOp, visited, columnNamesPerSuccessor);
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

                std::vector<mlir::daphne::ColumnProjectOp> columnProjectOps;
                std::vector<mlir::daphne::ConstantOp> columnNamesOps;

                for (auto columnName : columnNames) {
                    auto colName = rewriter.create<mlir::daphne::ConstantOp>(filterOp->getLoc(), columnName);
                    columnNamesOps.push_back(colName);
                    auto extract = rewriter.create<mlir::daphne::ExtractColOp>(filterOp->getLoc(), resTypeFrame, filterOp->getOperand(0), colName);
                    auto cast = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resType, extract->getResult(0));
                    columnProjectOps.push_back(rewriter.create<mlir::daphne::ColumnProjectOp>(filterOp->getLoc(), resType, cast->getResult(0), pos->getResult(0)));
                } 

                std::vector<mlir::Type> colTypes;
                std::vector<mlir::Value> cols;
                std::vector<mlir::Value> labels;

                //colTypes.push_back(matrix.getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());

                if(columnNames.size() == 1) {
                    auto castMatrix = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resTypeMatrix, columnProjectOps[0]->getResult(0));
                    cols.push_back(castMatrix->getResult(0));
                    labels.push_back(columnNamesOps[0]->getResult(0));
                    auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CreateFrameOp>(filterOp, resTypeFrame, cols, labels);
                    auto uses = filterOp->getUses();
                    for (auto x = uses.begin(); x != uses.end(); x++) {
                        x->getOwner()->replaceUsesOfWith(filterOp, res);
                    }
                    /**for (auto indexedResult : llvm::enumerate(filterOp->getResults())) {
                        Value result = indexedResult.value();
                        for (Operation *userOp : result.getUsers()) {
                            userOp->setOperand(0, res->getResult(0));
                        }
                    } **/
                    //filterOp.replaceAllUsesWith(res->getResult(0));
                    //auto uses = filterOp->getUses();
                    /**for (auto x = uses.begin(); uses; uses) {
                        std::cout << x.getOperandNumber() << std::endl;
                    } **/
                    //filterOp->erase();
                } else {
                    // TODO: directly create Frame with all columns
                    for (size_t i = 0; i < columnNames.size(); i++) {
                        auto castMatrix = rewriter.create<mlir::daphne::CastOp>(filterOp->getLoc(), resTypeMatrix, columnProjectOps[i]->getResult(0));
                        cols.push_back(castMatrix->getResult(0));
                        labels.push_back(columnNamesOps[i]->getResult(0));
                        colTypes.push_back(castMatrix->getResult(0).getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());
                    }

                    auto res = rewriter.replaceOpWithNewOp<mlir::daphne::CreateFrameOp>(filterOp, colTypes, cols, labels);
                    auto uses = filterOp->getUses();
                    for (auto x = uses.begin(); x != uses.end(); x++) {
                        x->getOwner()->replaceUsesOfWith(filterOp, res);
                    }
                } 
                
                
                /**WalkResult result = filterOp->walk([&](mlir::daphne::ConstantOp constantOp) {
                    if (!constantOp)
                        return WalkResult::interrupt();
                    mlir::Type constantType = constantOp->getResult(0).getType();
                    if (!constantType.isa<mlir::daphne::StringType>())
                        return WalkResult::advance();
                    columnNames.push_back(constantOp.getValue().cast<mlir::StringAttr>().getValue().str());
                    return WalkResult::advance();
                });
                if(result.wasInterrupted()) {
                    std::cout << "Test" << std::endl;
                }
                for (auto columnName : columnNames) {
                    std::cout << columnName << std::endl;
                }**/

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

