#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>

using namespace mlir;

uint32_t generateUniqueID() {
    static uint32_t currentID = 1;
    return currentID++;
}

class RecordPropertiesPass : public PassWrapper<RecordPropertiesPass, OperationPass<func::FuncOp>> {
public:
    
    RecordPropertiesPass() = default;
    
    StringRef getArgument() const final { return "record-properties"; }
    StringRef getDescription() const final { return "Record properties of different operations"; }
    
    void runOnOperation() override {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        func.walk([&](Operation *op) {
            if (isa<daphne::RecordPropertiesOp>(op) || isa<func::FuncOp>(op) || op->hasAttr("daphne.value_ids"))
                return;

            SmallVector<Attribute, 4> valueIDs;
            for (Value result : op->getResults()) {
                if (result.getType().isa<daphne::MatrixType>()) {
                    uint32_t id = generateUniqueID();
                    valueIDs.push_back(builder.getUI32IntegerAttr(id));

                    builder.setInsertionPointAfter(op);

                    auto idConstant = builder.create<mlir::daphne::ConstantOp>(
                        op->getLoc(),
                        builder.getIntegerType(32, /*isSigned=*/false),
                        builder.getUI32IntegerAttr(id)
                    );

                    builder.create<daphne::RecordPropertiesOp>(
                        op->getLoc(), result, idConstant
                    );
                }
            }

            if (!valueIDs.empty()) {
                op->setAttr("daphne.value_ids", builder.getArrayAttr(valueIDs));
            }
        });
    }
};

std::unique_ptr<Pass> daphne::createRecordPropertiesPass() {
    return std::make_unique<RecordPropertiesPass>();
}