/*
 * Copyright 2025 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <ir/daphneir/Passes.h>
 #include <ir/daphneir/Daphne.h>
 
#include <mlir/IR/Builders.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Visitors.h>

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
            
            if (auto castOp = dyn_cast<daphne::CastOp>(op)) {
                if (castOp.isRemovePropertyCast()) {
                    return WalkResult::advance();
                }
            }

            // Handle loops (scf.for and scf.while) and If blocks as black boxes
            if (isa<scf::ForOp>(op) || isa<scf::WhileOp>(op) || isa<scf::IfOp>(op)) {
                recordResults(op);
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