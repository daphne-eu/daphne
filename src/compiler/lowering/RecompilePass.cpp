/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cmath>
#include <memory>
#include <string>

#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <util/ErrorHandler.h>

namespace mlir {
#define GEN_PASS_DECL_RECOMPILEPASS
#define GEN_PASS_DEF_RECOMPILEPASS
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct RecompilePass : public impl::RecompilePassBase<RecompilePass> {

  public:
    explicit RecompilePass() : impl::RecompilePassBase<RecompilePass>() {}
    void runOnOperation() override;
};
} // end anonymous namespace

void wrapLoopBodyWithRecompileOp(mlir::scf::ForOp forOp, MLIRContext *context) {
    std::string loopBodyIR;
    SmallVector<Value> loopBodyInputValues;

    // Mapping for external objects -> loop body uses
    llvm::DenseMap<Value, SmallVector<mlir::Operation *, 4>> externalObjectUses;

    // Step 1: Collect inputs and map external objects to their uses
    // Add iter_args to loopBodyInputValues (already mapped)
    for (Value arg : forOp.getRegionIterArgs()) {
        if (std::find(loopBodyInputValues.begin(), loopBodyInputValues.end(), arg) == loopBodyInputValues.end()) {
            loopBodyInputValues.push_back(arg);
        }
    }

    size_t iter_args_size = loopBodyInputValues.size();

    // Map external uses
    forOp.getBody()->walk([&](mlir::Operation *op) {
        for (Value operand : op->getOperands()) {
            if (operand.getDefiningOp() &&
                operand.getDefiningOp()->getParentRegion() != forOp.getBody()->getParent()) {
                if (std::find(loopBodyInputValues.begin(), loopBodyInputValues.end(), operand) == loopBodyInputValues.end()) {
                    loopBodyInputValues.push_back(operand);
                }
                externalObjectUses[operand].push_back(op);
            }
        }
    });

    OpBuilder builder(forOp);
    auto resultTypes = forOp.getResultTypes();

    // Step 2: Create function inputs (types and values)
    SmallVector<Value> inputs;
    SmallVector<Type> inputTypes;
    SmallVector<bool> isScalarInput;

    for (Value &extVal : loopBodyInputValues) {
        if (!llvm::isa<daphne::MatrixType, daphne::FrameType, daphne::ListType, daphne::StringType>(extVal.getType())) {
            auto matType = mlir::daphne::MatrixType::get(context, extVal.getType(), 1, 1, -1, mlir::daphne::MatrixRepresentation::Default);
            inputTypes.push_back(matType);
            isScalarInput.push_back(true);

            mlir::Value ins = static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(extVal.getLoc(), matType, extVal));
            inputs.push_back(ins);
        } else {
            inputTypes.push_back(extVal.getType());
            inputs.push_back(extVal);
            isScalarInput.push_back(false);
        }
    }

    auto funcType = FunctionType::get(context, inputTypes, resultTypes);
    mlir::OwningOpRef<mlir::ModuleOp> tempModule = mlir::ModuleOp::create(forOp.getLoc());
    OpBuilder moduleBuilder(tempModule->getBodyRegion());
    auto funcOp = moduleBuilder.create<mlir::func::FuncOp>(forOp.getLoc(), "main", funcType);
    funcOp.getBody().takeBody(forOp.getLoopBody());

    // Step 3: Delete the Index argument and add arguments to the function's entry block
    Block &entryBlock = funcOp.getBody().front();

    // Delete the index argument and remove its CastOp
    BlockArgument argToRemove = entryBlock.getArgument(0);

    if (!argToRemove.use_empty()) {
        auto constantValue = builder.create<mlir::daphne::ConstantOp>(
            funcOp.getLoc(),
            builder.getIndexType(),
            builder.getIntegerAttr(builder.getIndexType(), 0)
        );

        argToRemove.replaceAllUsesWith(constantValue);
    }
    entryBlock.eraseArgument(0);

    funcOp.walk([&](mlir::Operation *op) {
        // Find the daphne.cast operation defining %0
        if (auto castOp = dyn_cast<mlir::daphne::CastOp>(op)) {
            Value result = castOp.getResult();
            
            // Replace all uses of %0 with a constant value
            if (!result.use_empty()) {
                auto replacementValue = builder.create<mlir::daphne::ConstantOp>(
                    op->getLoc(),
                    builder.getIntegerType(64),
                    builder.getI64IntegerAttr(10) // Example replacement
                );
                result.replaceAllUsesWith(replacementValue);
            }

            // Erase the cast operation
            castOp.erase();
        }
    });

    // Add external inputs
    for(size_t i = iter_args_size; i < inputTypes.size() ; i++) {
        entryBlock.addArgument(inputTypes[i], funcOp.getLoc());
    }

    // Step 4: Replace external object uses with corresponding arguments
    funcOp.walk([&](mlir::Operation *op) {
        for (auto &operand : op->getOpOperands()) {
            Value extObject = operand.get();
            if (externalObjectUses.count(extObject)) {
                unsigned index = std::distance(
                    loopBodyInputValues.begin(),
                    std::find(loopBodyInputValues.begin(), loopBodyInputValues.end(), extObject)
                );
                operand.set(entryBlock.getArgument(index));
            }
        }
    });

    // Step 4.1: Cast back the 1x1 matrices, how should be scalars and update their uses
    for (size_t i = 0; i < inputTypes.size(); ++i) {
        if (isScalarInput[i]) {
            auto arg = entryBlock.getArgument(i);
            auto scalarType = loopBodyInputValues[i].getType();
            builder.setInsertionPointToStart(&entryBlock);
            auto castToScalar = builder.create<mlir::daphne::CastOp>(
                funcOp.getLoc(), scalarType, arg);
            for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
                if (use.getOwner() != castToScalar.getOperation()) {
                    use.getOwner()->setOperand(use.getOperandNumber(), castToScalar);
                }
            }
        }
    }

    // Step 5: Replace `scf.yield` with `func.return`
    funcOp.walk([&](mlir::Operation *op) {
        if (isa<mlir::scf::YieldOp>(op)) {
            builder.setInsertionPoint(op);
            builder.create<mlir::func::ReturnOp>(op->getLoc(), op->getOperands());
            op->erase();
        }
    });

    // Debug: Print the transformed function
    llvm::raw_string_ostream os(loopBodyIR);
    tempModule->print(os);
    os.flush();

    llvm::errs() << "Transformed Function:\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n";

    // Step 6: Create and replace the RecompileOp
    auto stringType = builder.getType<mlir::daphne::StringType>();
    auto irStringAttr = builder.getStringAttr(loopBodyIR);
    auto irString = builder.create<mlir::daphne::ConstantOp>(forOp.getLoc(), stringType, irStringAttr).getResult();

    auto numInputs = builder.create<mlir::daphne::ConstantOp>(
        forOp.getLoc(), builder.getIntegerType(64), builder.getI64IntegerAttr(inputs.size())).getResult();

    SmallVector<Value> outputs(forOp.getResults().begin(), forOp.getResults().end());

    auto recompileOp = builder.create<mlir::daphne::RecompileOp>(
        forOp.getLoc(),
        irString,
        numInputs,
        inputs,
        outputs
    );

    // Replace ForOp results with RecompileOp results and erase for Op
    for (auto result : llvm::zip(forOp.getResults(), recompileOp.getResults())) {
        auto originalResult = std::get<0>(result);
        auto newResult = std::get<1>(result);
        originalResult.replaceAllUsesWith(newResult);
    }

    forOp.getBody()->erase();
}

void RecompilePass::runOnOperation() {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    func.walk([&](mlir::Operation *op) {
        if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
            wrapLoopBodyWithRecompileOp(forOp, context);
        }
    });
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createRecompilePass() {
    return std::make_unique<RecompilePass>();
}