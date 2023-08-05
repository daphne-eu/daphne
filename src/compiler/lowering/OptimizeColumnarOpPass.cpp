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
    void dfsIntersect(Operation *op, std::vector<Operation *> &intersects) {
        for (Operation *intersectFollowOp : op->getResult(0).getUsers()) {
            if (llvm::dyn_cast<mlir::daphne::ColumnIntersectOp>(intersectFollowOp)) {
                intersects.push_back(op);
                dfsIntersect(intersectFollowOp, intersects);
            }
        }
    }

    void insertBetween(mlir::OpBuilder &builder, Operation *op) {
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
        } else {
            std::cout << std::boolalpha <<"Intersect" << intersectToRemove->getResult(0).getUsers().empty() << std::endl;
            for (Operation *intersectToRemoveFollowOp : intersectToRemove->getResult(0).getUsers()) {
                intersectToRemoveFollowOp->replaceUsesOfWith(intersectToRemove->getResult(0), intersectToRemovePrevOp->getResult(0));
            }
            auto intersectLeftOp = intersectToRemove->getOperand(0).getDefiningOp();
            auto intersectRightOp = intersectToRemove->getOperand(1).getDefiningOp();
            if (intersectLeftOp == leOp || intersectRightOp == geOp) {
                commonIntersectOp->replaceUsesOfWith(leOp->getResult(0), betweenOp->getResult(0));
            } else {
                commonIntersectOp->replaceUsesOfWith(geOp->getResult(0), betweenOp->getResult(0));
            }
        }
    }

//projectionPath (evtl updateProjectionPath extra)
    void insertProjectionPath(mlir::OpBuilder &builder, Operation *op) {
        mlir::Type vt = mlir::daphne::UnknownType::get(builder.getContext());
        mlir::Type resType = mlir::daphne::ColumnType::get(
            builder.getContext(), vt
        );
        auto projectResult = op->getResult(0);
        for (Operation * projectFollowOp : projectResult.getUsers()) {
            auto projectFollowOpResult = projectFollowOp->getResult(0);
            if (llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(projectFollowOp)) {
                builder.setInsertionPointAfter(projectFollowOp);
                std::vector<mlir::Value> posLists{projectFollowOp->getOperand(1), op->getOperand(1)};
                auto projectionPathOp = builder.create<daphne::ColumnProjectionPathOp>(builder.getUnknownLoc(), resType, op->getOperand(0), posLists);
            
                for (Operation *projectTwoFollowOp : projectFollowOpResult.getUsers()) {
                    projectTwoFollowOp->replaceUsesOfWith(projectFollowOpResult, projectionPathOp->getResult(0));
                }
            }
        }
    }

    struct OptimizeColumnarOpPass : public PassWrapper<OptimizeColumnarOpPass, OperationPass<func::FuncOp>> {

        OptimizeColumnarOpPass() = default;
    
        void runOnOperation() final;
    
    };
}

void OptimizeColumnarOpPass::runOnOperation() {
    func::FuncOp f = getOperation();
    
    f->walk([&](Operation *op) {
        if(llvm::dyn_cast<mlir::daphne::ColumnLeOp>(op)) {
            OpBuilder builder(op);
            insertBetween(builder, op);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnGeOp>(op)) {
            OpBuilder builder(op);
            insertBetween(builder, op);
        } else if(llvm::dyn_cast<mlir::daphne::ColumnProjectOp>(op)) {
            OpBuilder builder(op);
            insertProjectionPath(builder, op);
        }
    });    
}

std::unique_ptr<Pass> daphne::createOptimizeColumnarOpPass()
{
    return std::make_unique<OptimizeColumnarOpPass>();
}

