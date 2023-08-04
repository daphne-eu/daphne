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
#include <vector>

using namespace mlir;

namespace
{
    void eraseOps(std::vector<Operation*> ops){
        for(auto op : ops){
            if(!op->use_empty()) {
                ops.push_back(op);
                continue;
            }
            op->erase();
        }
    }

    void checkAndRemoveCmpCasts(mlir::OpBuilder &builder, Operation *op, std::vector<Operation *> &toDelete) {
        mlir::Type vt = mlir::daphne::UnknownType::get(builder.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            builder.getContext(), vt
        );
        auto indexedResult = op->getResult(0);
        for (Operation *userOp : indexedResult.getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::CastOp>(userOp)) {
                auto resultCast = userOp->getResult(0);
                for (Operation *followCastOp : resultCast.getUsers()) {
                    if(llvm::dyn_cast<mlir::daphne::PositionListBitmapConverterOp>(followCastOp)) {
                        auto numRowsOp = followCastOp->getOperand(1).getDefiningOp();
                        auto resultBitmap = followCastOp->getResult(0);
                        for (Operation *followBitmapOp : resultBitmap.getUsers()) {
                            if (llvm::dyn_cast<mlir::daphne::CreateFrameOp>(followBitmapOp)) {
                                auto frameResult = followBitmapOp->getResult(0);
                                for (Operation *followFrameOp : frameResult.getUsers()) {
                                    if (llvm::dyn_cast<mlir::daphne::CastOp>(followFrameOp)) {
                                        auto frameCastResult = followFrameOp->getResult(0);
                                        for (Operation *followFrameCastOp : frameCastResult.getUsers()){
                                            if (llvm::dyn_cast<mlir::daphne::ColumnAndOp>(followFrameCastOp)) {
                                                builder.setInsertionPointAfter(followFrameCastOp);
                                                auto intersect = builder.create<daphne::ColumnIntersectOp>(builder.getUnknownLoc(), resType, followFrameCastOp->getOperands());
                                                intersect->replaceUsesOfWith(frameCastResult, indexedResult);
                                                auto andResult = followFrameCastOp->getResult(0);
                                                for (Operation *andSuccessorOp : andResult.getUsers()) {
                                                    andSuccessorOp->replaceUsesOfWith(andResult, intersect->getResult(0));
                                                }
                                                eraseOps({followFrameCastOp, followFrameOp, followBitmapOp, followCastOp, numRowsOp});
                                                toDelete.push_back(userOp);
                                            } else if(llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(followFrameCastOp)) {
                                                followFrameCastOp->replaceUsesOfWith(frameCastResult, indexedResult);
                                                eraseOps({followFrameOp, followBitmapOp, followCastOp, numRowsOp});
                                                toDelete.push_back(userOp);
                                            } else if(llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(followFrameCastOp)) {
                                                followFrameCastOp->replaceUsesOfWith(frameCastResult, indexedResult);
                                                eraseOps({followFrameOp, followBitmapOp, followCastOp, numRowsOp});
                                                toDelete.push_back(userOp);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void checkAndRemoveIntersectCasts(mlir::OpBuilder &builder, Operation *op, std::vector<Operation *> &toDelete) {
        auto indexedResult = op->getResult(0);
        for (Operation *userOp : indexedResult.getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::CastOp>(userOp)) {
                for (Operation *followCastOp : userOp->getResult(0).getUsers()){
                    if (llvm::dyn_cast<mlir::daphne::BitmapPositionListConverterOp>(followCastOp)) {
                        for (Operation *followBitmapOp : followCastOp->getResult(0).getUsers()) {
                            if (llvm::dyn_cast<mlir::daphne::CastOp>(followBitmapOp)) {
                                auto bitmapCastResult = followBitmapOp->getResult(0);
                                std::vector<Operation *> toReplace;
                                for (Operation *followBitmapCastOp : bitmapCastResult.getUsers()) {
                                    if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(followBitmapCastOp)) {
                                        toReplace.push_back(followBitmapCastOp);
                                    }
                                }
                                for (Operation *replaceUsesOp : toReplace) {
                                    replaceUsesOp->replaceUsesOfWith(bitmapCastResult, indexedResult);
                                }
                                eraseOps({followBitmapOp, followCastOp});
                                toDelete.push_back(userOp);
                            }
                        }
                    }
                }
            }
        }
    }

    void checkAndRemoveCreateFrames(mlir::OpBuilder &builder, Operation *op, std::vector<Operation *> &toDelete) {
        bool deleteFrameOp = false;
        std::vector<Operation *> pushToDelete;
        auto numInputColMatrix = op->getNumOperands()/2;
        std::map<std::string, Operation *> inputColumns;
        std::map<std::string, std::pair<Operation *, Operation *>> outputColumns;
        for (int i = 0; i < numInputColMatrix; i++) {
            auto inputName = op->getOperand(i+numInputColMatrix).getDefiningOp()->getAttr("value").cast<StringAttr>().getValue().str();
            auto inputMatrixOp = op->getOperand(i).getDefiningOp();
            if (llvm::dyn_cast<mlir::daphne::CastOp>(inputMatrixOp)) {
                auto inputColumnOp = inputMatrixOp->getOperand(0).getDefiningOp();
                if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(inputColumnOp)) {
                    inputColumns[inputName] = inputColumnOp;
                    pushToDelete.push_back(inputMatrixOp);
                }
            }
        }
        auto frameResult = op->getResult(0);
        for (Operation *followFrameOp : frameResult.getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::ExtractColOp>(followFrameOp)) {
                auto extractName = followFrameOp->getOperand(1).getDefiningOp()->getAttr("value").cast<StringAttr>().getValue().str();
                auto extractResult = followFrameOp->getResult(0);
                for (Operation *followExtractOp : extractResult.getUsers()) {
                    if (llvm::dyn_cast<mlir::daphne::CastOp>(followExtractOp)) {
                        auto extractCastResult = followExtractOp->getResult(0);
                        for (Operation *followExtractCastOp : extractCastResult.getUsers()) {
                            outputColumns[extractName] = std::make_pair(followExtractCastOp, followExtractOp);
                            toDelete.push_back(followExtractOp);
                            toDelete.push_back(followFrameOp);
                            deleteFrameOp = true;
                        }
                    }
                }
            }
        }

        if(deleteFrameOp) {
            toDelete.push_back(op);
            toDelete.insert(toDelete.end(), pushToDelete.begin(), pushToDelete.end());
        }

        for (auto columnIn = inputColumns.begin(); columnIn != inputColumns.end(); columnIn++) {
            std::string inputColName = columnIn->first;
            Operation *inputColOp = columnIn->second;
            for (auto columnOut = outputColumns.begin(); columnOut != outputColumns.end(); columnOut++) {
                std::string outputColName = columnOut->first;
                Operation *outputColOp = columnOut->second.first;
                Operation *replaceCastOp = columnOut->second.second;
                if (inputColName == outputColName) {
                    outputColOp->replaceUsesOfWith(replaceCastOp->getResult(0), inputColOp->getResult(0));
                }
            }
        }
    }

    struct ReduceColumnarOpPass : public PassWrapper<ReduceColumnarOpPass, OperationPass<func::FuncOp>> {

        ReduceColumnarOpPass() = default;
    
        void runOnOperation() final;
    
    };
}

void ReduceColumnarOpPass::runOnOperation() {
    func::FuncOp f = getOperation();
    std::vector<Operation *> toDelete;
    f->walk([&](Operation *op) {

        if(llvm::dyn_cast<mlir::daphne::ColumnGeOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnGtOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnLeOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnLtOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnEqOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnNeqOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCmpCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveIntersectCasts(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::CreateFrameOp>(op)) {
            OpBuilder builder(op);
            checkAndRemoveCreateFrames(builder, op, toDelete);
        }
    });

    eraseOps(toDelete);
    
}

std::unique_ptr<Pass> daphne::createReduceColumnarOpPass()
{
    return std::make_unique<ReduceColumnarOpPass>();
}

