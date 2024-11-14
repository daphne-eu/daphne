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
    SmallVector<Value> externalValues;

    forOp.getBody()->walk([&](mlir::Operation *op) {
        for (Value operand : op->getOperands()) {
            if (operand.getDefiningOp() && operand.getDefiningOp()->getParentRegion() != forOp.getBody()->getParent()) {
                if (std::find(externalValues.begin(), externalValues.end(), operand) == externalValues.end()) {
                    externalValues.push_back(operand);
                }
            }
        }
    });
    
    auto resultTypes = forOp.getResultTypes();
    {
        mlir::OwningOpRef<mlir::ModuleOp> tempModule = mlir::ModuleOp::create(forOp.getLoc());
        OpBuilder moduleBuilder(tempModule->getBodyRegion());

        SmallVector<Type> inputTypes;
        for (Value arg : forOp.getRegionIterArgs()) {
            inputTypes.push_back(arg.getType());
        }
        for (Value extVal : externalValues) {
            inputTypes.push_back(extVal.getType());
        }

        
        auto funcType = FunctionType::get(context, inputTypes, resultTypes);

        auto funcOp = moduleBuilder.create<mlir::func::FuncOp>(forOp.getLoc(), "loop_body", funcType);

        funcOp.getBody().takeBody(forOp.getLoopBody());

        llvm::raw_string_ostream os(loopBodyIR);
        tempModule->print(os);
        os.flush();
    }

    OpBuilder builder(forOp);
    auto stringType = builder.getType<mlir::daphne::StringType>();
    auto irStringAttr = builder.getStringAttr(loopBodyIR);
    auto irString = builder.create<mlir::daphne::ConstantOp>(forOp.getLoc(), stringType, irStringAttr).getResult();

    SmallVector<Value> inputs;
    inputs.append(externalValues.begin(), externalValues.end()); 

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

    for (auto result : llvm::zip(forOp.getResults(), recompileOp.getResults())) {
        std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
    }

    forOp.erase();
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