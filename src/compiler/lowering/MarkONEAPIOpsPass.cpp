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
#ifdef USE_ONEAPI
#include "compiler/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <mlir/IR/BlockAndValueMapping.h>

#include <iostream>

using namespace mlir;

struct MarkONEAPIOpsPass : public PassWrapper<MarkONEAPIOpsPass, FunctionPass> {
    
    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;
    
    explicit MarkONEAPIOpsPass(const DaphneUserConfig& cfg) : cfg(cfg) {
    }
    
    void runOnFunction() final;
    
    bool checkUseONEAPI(Operation* op) const {
//        std::cout << "checkUseONEAPI: " << op->getName().getStringRef().str() << std::endl;
        return op->hasTrait<mlir::OpTrait::ONEAPISupport>();
    }
};

void MarkONEAPIOpsPass::runOnFunction() {
    getFunction()->walk([&](Operation* op) {
        OpBuilder builder(op);
        if(checkUseONEAPI(op)) {
            op->setAttr("oneapi_device", builder.getI32IntegerAttr(0));
        }
        WalkResult::advance();
    });
}

std::unique_ptr<Pass> daphne::createMarkONEAPIOpsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<MarkONEAPIOpsPass>(cfg);
}
#endif // USE_ONEAPI