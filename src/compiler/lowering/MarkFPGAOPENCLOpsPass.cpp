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
#ifdef USE_FPGAOPENCL
#include "compiler/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <mlir/IR/BlockAndValueMapping.h>

#include <iostream>

using namespace mlir;

struct MarkFPGAOPENCLOpsPass : public PassWrapper<MarkFPGAOPENCLOpsPass, FunctionPass> {

    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;

    explicit MarkFPGAOPENCLOpsPass(const DaphneUserConfig& cfg) : cfg(cfg) {
    }

    void runOnFunction() final;

    bool checkUseFPGAOPENCL(Operation* op) const {
//        std::cout << "checkUseFPGAOPENCL: " << op->getName().getStringRef().str() << std::endl;
        return op->hasTrait<mlir::OpTrait::FPGAOPENCLSupport>();
    }
};

void MarkFPGAOPENCLOpsPass::runOnFunction() {
    getFunction()->walk([&](Operation* op) {
        OpBuilder builder(op);
        if(checkUseFPGAOPENCL(op)) {
            op->setAttr("fpgaopencl_device", builder.getI32IntegerAttr(0));
        }
        WalkResult::advance();
    });
}

std::unique_ptr<Pass> daphne::createMarkFPGAOPENCLOpsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<MarkFPGAOPENCLOpsPass>(cfg);
}
#endif 
