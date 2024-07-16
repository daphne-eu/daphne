#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"
#include <sstream>
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>

using namespace mlir;


// Utility to generate a unique ID string
std::string generateUniqueID() {
    static int64_t currentID = 0;
    std::stringstream ss;
    ss << "op_id_" << currentID++;
    return ss.str();
}

// Function to capture properties and set them as attributes on the operation
void captureProperties(Operation *op, OpBuilder &builder) {
    // Example: Capture type information
    Type resultType = op->getResult(0).getType();
    op->setAttr("daphne.result_type", TypeAttr::get(resultType));

    // Capture shape using the DaphneInferShapeOpInterface
    if (auto inferShapeOp = llvm::dyn_cast<daphne::InferShape>(op)) {
        auto shapes = inferShapeOp.inferShape();
        if (!shapes.empty()) {
            SmallVector<Attribute, 4> shapeAttrs;
            for (const auto &shape : shapes) {
                shapeAttrs.push_back(builder.getI64ArrayAttr({shape.first, shape.second}));
            }
            ArrayAttr shapeArrayAttr = builder.getArrayAttr(shapeAttrs);
            op->setAttr("daphne.shape", shapeArrayAttr);
        }
    }

    // Capture sparsity using the DaphneInferSparsityOpInterface
    if (auto inferSparsityOp = llvm::dyn_cast<daphne::InferSparsity>(op)) {
        auto sparsities = inferSparsityOp.inferSparsity();
        if (!sparsities.empty()) {
            SmallVector<Attribute, 4> sparsityAttrs;
            for (auto sparsity : sparsities) {
                sparsityAttrs.push_back(builder.getF64FloatAttr(sparsity));
            }
            ArrayAttr sparsityArrayAttr = builder.getArrayAttr(sparsityAttrs);
            op->setAttr("daphne.sparsity", sparsityArrayAttr);
        }
    }
}

// Pass to record properties and assign unique IDs to operations
class RecordPropertiesPass : public PassWrapper<RecordPropertiesPass, OperationPass<func::FuncOp>> {
public:
    StringRef getArgument() const final { return "record-properties"; }
    StringRef getDescription() const final { return "Records properties of operations."; }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        func.walk([&](Operation *op) {
            // Skip the RecordOp itself to avoid recursion
            if (isa<daphne::RecordOp>(op))
                return;

            // Set unique ID if not already set
            if (!op->hasAttr("daphne.id")) {
                std::string id = generateUniqueID();
                op->setAttr("daphne.id", builder.getStringAttr(id));
            }

            // Record properties
            for (Value result : op->getResults()) {
                builder.setInsertionPointAfter(op);
                auto recordOp = builder.create<daphne::RecordOp>(op->getLoc(), result);
                recordOp->setOperand(0, result);
                recordOp->getResult(0).setType(result.getType());

                // Propagate the ID to RecordOp
                recordOp->setAttr("daphne.id", op->getAttr("daphne.id"));

                // Capture and propagate additional properties
                captureProperties(recordOp.getOperation(), builder);
            }
        });
    }
};

std::unique_ptr<Pass> daphne::createRecordPropertiesPass() {

    return std::make_unique<RecordPropertiesPass>();
}