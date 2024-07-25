#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>

using namespace mlir;


int64_t generateUniqueID() {
    static int64_t currentID = 1;
    return currentID++;
}

class RecordPropertiesPass : public PassWrapper<RecordPropertiesPass, OperationPass<func::FuncOp>> {
public:

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        func.walk([&](Operation *op) {
            if (isa<daphne::RecordPropertiesOp>(op) || isa<func::FuncOp>(op))
                return;

            bool hasMatrixResult = false;
            for (Value result : op->getResults()) {
                if (result.getType().isa<daphne::MatrixType>()) {
                    hasMatrixResult = true;
                    break;
                }
            }

            if (hasMatrixResult && !op->hasAttr("daphne.op_id")) {
                int64_t id = generateUniqueID();
                op->setAttr("daphne.op_id", builder.getI64IntegerAttr(id));
                
                for (Value result: op->getResults()){  
                    if (result.getType().isa<daphne::MatrixType>()) {
                        builder.setInsertionPointAfter(op);

                        auto idConstant = builder.create<mlir::daphne::ConstantOp>(
                            op->getLoc(), 
                            builder.getIntegerType(64, /*isSigned=*/true), 
                            builder.getI64IntegerAttr(id)
                        );

                        builder.create<daphne::RecordPropertiesOp>(
                            op->getLoc(), result, idConstant
                        );

                    }    
                }
            }
        });   
    }
};

std::unique_ptr<Pass> daphne::createRecordPropertiesPass() {
    return std::make_unique<RecordPropertiesPass>();
}