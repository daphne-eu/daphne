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
    BlockArgument indexArg = entryBlock.getArgument(0);
     if (!indexArg.use_empty()) {    
        auto constantValue = tempBuilder.create<mlir::daphne::ConstantOp>(
            funcOp.getLoc(),
            tempBuilder.getIndexType(),
            tempBuilder.getIntegerAttr(tempBuilder.getIndexType(), 0)
        );
        
        indexArg.replaceAllUsesWith(constantValue);
    }
    entryBlock.eraseArgument(0);
    
    for(size_t i = iter_args_size; i < inputTypes.size() ; i++) {
        entryBlock.addArgument(inputTypes[i], funcOp.getLoc());
    }
    
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
    
    // Create New For Op
    builder.setInsertionPoint(forOp);

    Location loc = forOp.getLoc();
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();
    SmallVector<Value> iterArgs(forOp.getIterOperands().begin(), forOp.getIterOperands().end());
    auto newForOp = builder.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, iterArgs);

    Block *newBlock = newForOp.getBody();

    builder.setInsertionPointToStart(newBlock);

    // Create Induction Var Cast
    /** 
    Value inductionVar = newForOp.getInductionVar();
    builder.create<mlir::daphne::CastOp>(
    loc, builder.getI64Type(), inductionVar);
    */

    // Create Loop Body String
    auto irString = builder.create<mlir::daphne::ConstantOp>(
        newForOp.getLoc(), 
        builder.getType<mlir::daphne::StringType>(), 
        builder.getStringAttr(loopBodyIR)).getResult();
    
    // Create RecompileOp with new args
    Block *loopBody = newForOp.getBody();
    BlockArgument arg1 = loopBody->getArgument(1);
    BlockArgument arg2 = loopBody->getArgument(2);
    inputs[0] = arg1;
    inputs[1] = arg2;

    auto recompileOp = builder.create<mlir::daphne::RecompileOp>(
        forOp.getLoc(),
        resultTypes,
        inputs,
        irString
    );

    //Rewire Recompile -> Yield and Yield -> old for loop results
    SmallVector<Value> yieldValues(recompileOp.getResults().begin(), recompileOp.getResults().end());
    builder.create<mlir::scf::YieldOp>(loc, yieldValues);

    
    for (auto [oldResult, newResult] : llvm::zip(forOp.getResults(), newForOp.getResults())) {
        oldResult.replaceAllUsesWith(newResult);
    }

    forOp.erase();
}

void RecompilePass::runOnOperation() {
    auto func = getOperation();
    MLIRContext *context = &getContext();
 
    llvm::errs() << "Transformed IR before RecompilePass:\n";
    func.print(llvm::errs());
    llvm::errs() << "\n";
    

    func.walk([&](mlir::Operation *op) {
        if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
            wrapLoopBodyWithRecompileOp(forOp, context);
        }
    });
     
    llvm::errs() << "Transformed IR after RecompilePass:\n";
    func.print(llvm::errs());
    llvm::errs() << "\n";
    
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::daphne::createRecompilePass() {
    return std::make_unique<RecompilePass>();
}