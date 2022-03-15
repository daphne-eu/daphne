/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <compiler/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

#include <vector>

using namespace mlir;

/**
 * @brief Inserts a `IncRefKernelResOp` after each qualifying kernel call to
 * make sure that the reference counters of kernel inputs that were returned as
 * outputs are increased.
 * 
 * This is important to avoid double frees. 
 * 
 * A kernel call qualifies if at least one of its arguments is a data object
 * (matrix/frame) and at least one of its results is a data object.
 */
struct ManageObjRefsKernelResPass : public PassWrapper<ManageObjRefsKernelResPass, FunctionPass>
{
    explicit ManageObjRefsKernelResPass() {}
    void runOnFunction() final;
};

void ManageObjRefsKernelResPass::runOnFunction()
{
    getFunction().walk<WalkOrder::PreOrder>([](daphne::CallKernelOp cko){
        OpBuilder builder(cko.getContext());
        builder.setInsertionPointAfter(cko);
        std::vector<Value> objRess;
        std::vector<Value> objArgs;
        for(Value res : cko->getResults())
            if(CompilerUtils::hasObjType(res))
                objRess.push_back(res);
        for(Value arg : cko->getOperands())
            if(CompilerUtils::hasObjType(arg))
                objArgs.push_back(arg);
        if(objRess.size() && objArgs.size())
            builder.create<daphne::IncRefKernelResOp>(
                    cko.getLoc(), objRess, objArgs
            );
    });
}

std::unique_ptr<Pass> daphne::createManageObjRefsKernelResPass()
{
    return std::make_unique<ManageObjRefsKernelResPass>();
}