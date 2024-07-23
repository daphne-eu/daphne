#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include <sstream>
#include <ir/daphneir/Passes.h>
#include <ir/daphneir/Daphne.h>

using namespace mlir;

const char* generateUniqueID() {
    static int64_t currentID = 0;
    static std::vector<std::string> ids;

    std::stringstream ss;
    ss << "op_id_" << currentID++;
    ids.push_back(ss.str());

    return ids.back().c_str();
}

void captureProperties(Operation *op, OpBuilder &builder) {
    Type resultType = op->getResult(0).getType();
    op->setAttr("daphne.result_type", TypeAttr::get(resultType));

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

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        OpBuilder builder(func.getContext());

        func.walk([&](Operation *op) {
            if (isa<daphne::RecordOp>(op))
                return;

            if (!op->hasAttr("daphne.id")) {
                const char* id = generateUniqueID();
                op->setAttr("daphne.id", builder.getStringAttr(id));
            }

            for (Value result : op->getResults()) {
                builder.setInsertionPointAfter(op);
                auto recordOp = builder.create<daphne::RecordOp>(op->getLoc(), result);
                recordOp->setOperand(0, result);
                recordOp->getResult(0).setType(result.getType());

                recordOp->setAttr("daphne.id", op->getAttr("daphne.id"));

                captureProperties(recordOp.getOperation(), builder);
            }
        }        
        );
    }
};

std::unique_ptr<Pass> daphne::createRecordPropertiesPass() {
    return std::make_unique<RecordPropertiesPass>();
}