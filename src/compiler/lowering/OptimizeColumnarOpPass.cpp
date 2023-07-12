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

    void dfsIntersect(Operation *op, std::vector<Operation *> &intersects) {
        for (Operation *intersectFollowOp : op->getResult(0).getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(intersectFollowOp)) {
                intersects.push_back(op);
                dfsIntersect(intersectFollowOp, intersects);
            }
        }
    }

    void insertBetween(mlir::OpBuilder &builder, Operation *op, std::vector<Operation *> &toDelete) {
        mlir::Type vt = mlir::daphne::UnknownType::get(builder.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            builder.getContext(), vt
        );
        auto sourceDataOp = op->getOperand(0).getDefiningOp();
        auto sourceResult = sourceDataOp->getResult(0);
        Operation * leOp = nullptr;
        Operation * geOp = nullptr;
        for(Operation * sourceFollowOp : sourceResult.getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::ColumnLeOp>(sourceFollowOp)) {
                leOp = sourceFollowOp;
            } else if (llvm::dyn_cast<mlir::daphne::ColumnGeOp>(sourceFollowOp)) {
                geOp = sourceFollowOp;
            }
        }
        if(!geOp || !leOp) {
            return;
        }

        std::vector<Operation *> intersectsLeOp;
        std::vector<Operation *> intersectsGeOp;
        for (Operation* leFollowOp : leOp->getResult(0).getUsers()) {
            dfsIntersect(leFollowOp, intersectsLeOp);
        }
        for (Operation* geFollowOp : geOp->getResult(0).getUsers()) {
            dfsIntersect(geFollowOp, intersectsGeOp);
        }

        auto firstCommonIntersect = std::find_first_of (intersectsLeOp.begin(), intersectsLeOp.end(),
                               intersectsGeOp.begin(), intersectsGeOp.end());
        bool commonIntersect = firstCommonIntersect != intersectsLeOp.end();

        if (!commonIntersect) {
            return;
        }

        auto commonIntersectOp = *firstCommonIntersect;

        builder.setInsertionPointAfter(op);
        auto betweenOp = builder.create<daphne::ColumnBetweenOp>(builder.getUnknownLoc(), resType, sourceDataOp->getResult(0), geOp->getOperand(1), leOp->getOperand(1));

        Operation *intersectToRemove = nullptr;
        Operation *intersectToRemovePrevOp;
        for (Operation *leFollowOp : leOp->getResult(0).getUsers()) {
            if (mlir::dyn_cast<mlir::daphne::ColumnIntersectOp>(leFollowOp)) {
                if (leFollowOp == commonIntersectOp) {
                continue;
            }
            intersectToRemove = leFollowOp;
            Operation *intersectLeftOp = intersectToRemove->getOperand(0).getDefiningOp();
            Operation *intersectRightOp = intersectToRemove->getOperand(1).getDefiningOp();
            intersectToRemovePrevOp = intersectLeftOp == leOp ? intersectRightOp : intersectLeftOp;
            }
        }

        if(!intersectToRemove) {
            for (Operation *geFollowOp : geOp->getResult(0).getUsers()) {
                if (mlir::dyn_cast<mlir::daphne::ColumnIntersectOp>(geFollowOp)) {
                    if (geFollowOp == commonIntersectOp) {
                    continue;
                }
                intersectToRemove = geFollowOp;
                Operation *intersectLeftOp = intersectToRemove->getOperand(0).getDefiningOp();
                Operation *intersectRightOp = intersectToRemove->getOperand(1).getDefiningOp();
                intersectToRemovePrevOp = intersectLeftOp == geOp ? intersectRightOp : intersectLeftOp;
                }
            }
        }
        // They are both connected to the same intersect
        if(!intersectToRemove) {
            for (Operation *intersectFollowOp : commonIntersectOp->getResult(0).getUsers()) {
                intersectFollowOp->replaceUsesOfWith(commonIntersectOp->getResult(0), betweenOp->getResult(0));
            }
            toDelete.push_back(commonIntersectOp);
        } else {
            std::cout << std::boolalpha <<"Intersect" << intersectToRemove->getResult(0).getUsers().empty() << std::endl;
            for (Operation *intersectToRemoveFollowOp : intersectToRemove->getResult(0).getUsers()) {
                intersectToRemoveFollowOp->replaceUsesOfWith(intersectToRemove->getResult(0), intersectToRemovePrevOp->getResult(0));
            }
            toDelete.push_back(intersectToRemove);
            auto intersectLeftOp = intersectToRemove->getOperand(0).getDefiningOp();
            auto intersectRightOp = intersectToRemove->getOperand(1).getDefiningOp();
            if (intersectLeftOp == leOp || intersectRightOp == geOp) {
                commonIntersectOp->replaceUsesOfWith(leOp->getResult(0), betweenOp->getResult(0));
            } else {
                commonIntersectOp->replaceUsesOfWith(geOp->getResult(0), betweenOp->getResult(0));
            }
        }

        toDelete.push_back(leOp);
        toDelete.push_back(geOp);
    }

//projectionPath (evtl updateProjectionPath extra)
    void insertProjectionPath(mlir::OpBuilder &builder, Operation *op, std::vector<Operation *> &toDelete) {
        mlir::Type vt = mlir::daphne::UnknownType::get(builder.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            builder.getContext(), vt
        );
        auto projectResult = op->getResult(0);
        int successors = 0;
        int deleteCount = 0;
        for (Operation * projectFollowOp : projectResult.getUsers()) {
            auto projectFollowOpResult = projectFollowOp->getResult(0);
            if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(projectFollowOp)) {
                builder.setInsertionPointAfter(projectFollowOp);
                std::vector<mlir::Value> posLists{projectFollowOp->getOperand(1), op->getOperand(1)};
                auto projectionPathOp = builder.create<daphne::ColumnProjectionPathOp>(builder.getUnknownLoc(), resType, op->getOperand(0), posLists);
            
                for (Operation *projectTwoFollowOp : projectFollowOpResult.getUsers()) {
                    projectTwoFollowOp->replaceUsesOfWith(projectFollowOpResult, projectionPathOp->getResult(0));
                }

                if (projectFollowOp->use_empty()) {
                    toDelete.push_back(projectFollowOp);
                    deleteCount++;
                }
            }
            
            successors++;    
        }
        if (successors == deleteCount && successors > 0) {
            toDelete.push_back(op);
        }
    }

    struct OptimizeColumnarOpPass : public PassWrapper<OptimizeColumnarOpPass, OperationPass<func::FuncOp>> {

        OptimizeColumnarOpPass() = default;
    
        void runOnOperation() final;
    
    };
}

void OptimizeColumnarOpPass::runOnOperation() {
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
    toDelete.clear();

    f->walk([&](Operation *op) {
        if(llvm::dyn_cast<mlir::daphne::ColumnLeOp>(op)) {
            OpBuilder builder(op);
            insertBetween(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnGeOp>(op)) {
            OpBuilder builder(op);
            insertBetween(builder, op, toDelete);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(op)) {
            OpBuilder builder(op);
            insertProjectionPath(builder, op, toDelete);
        }
    });

    eraseOps(toDelete);
    
}

std::unique_ptr<Pass> daphne::createOptimizeColumnarOpPass()
{
    return std::make_unique<OptimizeColumnarOpPass>();
}

