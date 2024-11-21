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

    // Collect inputs and map external objects to their uses
    for (auto [regionArg, initArg] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
        // Use initArg instead of regionArg for proper linkage to the originating value
        if (std::find(loopBodyInputValues.begin(), loopBodyInputValues.end(), initArg) == loopBodyInputValues.end()) {
            loopBodyInputValues.push_back(initArg);
        }
    }

    size_t iter_args_size = loopBodyInputValues.size();

    // Map external uses: External objects -> Loop Body Uses
    llvm::DenseMap<Value, SmallVector<mlir::Operation *, 4>> externalObjectUses;
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

    // Create function inputs (types and values)
    SmallVector<Value> inputs;
    SmallVector<Type> inputTypes;
    SmallVector<bool> isScalarInput;
 
    for (Value extVal : loopBodyInputValues) {
        if (!llvm::isa<daphne::MatrixType, daphne::FrameType, daphne::ListType, daphne::StringType>(extVal.getType())) {
            auto matType = mlir::daphne::MatrixType::get(context, extVal.getType(), 1, 1, -1, mlir::daphne::MatrixRepresentation::Default);
            inputTypes.push_back(matType);
            isScalarInput.push_back(true);
            mlir::Value ins = builder.create<mlir::daphne::CastOp>(extVal.getLoc(), matType, extVal);
            inputs.push_back(ins);
        } else {
            inputTypes.push_back(extVal.getType());
            inputs.push_back(extVal);
            isScalarInput.push_back(false);
        }
    }
    
    auto funcType = FunctionType::get(context, inputTypes, resultTypes);
    mlir::OwningOpRef<mlir::ModuleOp> tempModule = mlir::ModuleOp::create(forOp.getLoc());
    
    OpBuilder tempBuilder(tempModule->getBodyRegion());
    auto funcOp = tempBuilder.create<mlir::func::FuncOp>(forOp.getLoc(), "main", funcType);
    funcOp.getBody().takeBody(forOp.getLoopBody());

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
 
    for(size_t i = iter_args_size; i < inputTypes.size() ; i++) {
        entryBlock.addArgument(inputTypes[i], funcOp.getLoc());
    }

    /** 
    for (size_t i = 0; i < entryBlock.getNumArguments(); ++i) {
        auto blockArg = entryBlock.getArgument(i);
        Value definingValue = loopBodyInputValues[i];
        if (!blockArg.use_empty() && definingValue) {
            llvm::errs() << "Replacing block argument " << i << " with SSA value:\n";
            llvm::errs() << "  Block Argument: ";
            blockArg.print(llvm::errs());
            llvm::errs() << "\n  SSA Value: ";
            definingValue.print(llvm::errs());
            llvm::errs() << "\n";

            blockArg.replaceAllUsesWith(definingValue);
        }
    }
    */

    // Replace external object uses with corresponding arguments
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

    // Cast back the 1x1 matrices, how should be scalars and update their uses
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

    // Replace `scf.yield` with `func.return`
    funcOp.walk([&](mlir::Operation *op) {
        if (isa<mlir::scf::YieldOp>(op)) {
            builder.setInsertionPoint(op);
            builder.create<mlir::func::ReturnOp>(op->getLoc(), op->getOperands());
            op->erase();
        }
    });

    llvm::raw_string_ostream os(loopBodyIR);
    tempModule->print(os);
    os.flush();
    
    builder.setInsertionPoint(forOp);
    auto irString = builder.create<mlir::daphne::ConstantOp>(
        forOp.getLoc(), 
        builder.getType<mlir::daphne::StringType>(), 
        builder.getStringAttr(loopBodyIR)).getResult();

    auto numInputs = builder.create<mlir::daphne::ConstantOp>(
        forOp.getLoc(),
        builder.getIntegerType(64),
        builder.getI64IntegerAttr(inputs.size())
    ).getResult();

    llvm::errs() << "Creating RecompileOp with operands:\n";
    llvm::errs() << " Result Types: " << resultTypes << "\n";
    llvm::errs() << "  IR String: " << irString << "\n";
    llvm::errs() << "  Num Inputs: " << numInputs << "\n";
    llvm::errs() << "  Inputs:\n";
    for (auto input : inputs) {
        llvm::errs() << "    Input: " << input << " Type: " << input.getType() << "\n";
    }

    auto recompileOp = builder.create<mlir::daphne::RecompileOp>(
        forOp.getLoc(),
        resultTypes,
        irString,
        numInputs,
        inputs
    );

    mlir::ValueTypeRange<mlir::OperandRange> operandTypes = recompileOp->getOperands();
    llvm::errs() << "RecompileOp Operand Count: " << operandTypes.size() << "\n";

    for (auto [forResult, recompileResult] : llvm::zip(forOp.getResults(), recompileOp.getResults())) {
        forResult.replaceAllUsesWith(recompileResult);
    }

    forOp.erase();
}

void RecompilePass::runOnOperation() {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    /** 
    llvm::errs() << "Transformed IR before RecompilePass:\n";
    func.print(llvm::errs());
    llvm::errs() << "\n";
    */

    func.walk([&](mlir::Operation *op) {
        if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
            wrapLoopBodyWithRecompileOp(forOp, context);
        }
    });

    /** 
    llvm::errs() << "Transformed IR after RecompilePass:\n";
    func.print(llvm::errs());
    llvm::errs() << "\n";
    */
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createRecompilePass() {
    return std::make_unique<RecompilePass>();
}