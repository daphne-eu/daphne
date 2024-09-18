#include <mlir/IR/Builders.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "ir/daphneir/Passes.h"
#include "ir/daphneir/Daphne.h"
#include "mlir/IR/Visitors.h"

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

        auto recordResults = [&](Operation *op) {
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
        };

        func.walk<WalkOrder::PreOrder>([&](Operation *op)-> WalkResult {
            // Skip specific ops that should not be processed
            if (isa<daphne::RecordPropertiesOp>(op) || op->hasAttr("daphne.value_ids"))
                return WalkResult::advance();
            
            // Handle loops (scf.for and scf.while) as black boxes
            else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
                recordResults(forOp);
                return WalkResult::skip();
            }
            else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
                recordResults(whileOp);
                return WalkResult::skip();
            }
            else if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
            // Check if this is the @main function or a UDF
                if (funcOp.getName() == "main") {
                    return WalkResult::advance();
                } else {
                    return WalkResult::skip();
                }
            }
            // Process other operations with Matrix Output type
            else {
                recordResults(op);
                return WalkResult::advance();
            }
        });
    }
};

std::unique_ptr<Pass> daphne::createRecordPropertiesPass() {
    return std::make_unique<RecordPropertiesPass>();
}