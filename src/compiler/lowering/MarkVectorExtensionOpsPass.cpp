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
#if defined USE_AVX512 || defined USE_AVX2 || defined USE_SSE || defined USE_SCALAR
#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "runtime/local/kernels/SIMDOperatorsDAPHNE/VectorExtensions.h"

#include <iostream>

using namespace mlir;

struct MarkVectorExtensionOpsPass : public PassWrapper<MarkVectorExtensionOpsPass, OperationPass<func::FuncOp>> {

    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;

    explicit MarkVectorExtensionOpsPass(const DaphneUserConfig& cfg) : cfg(cfg) {
    }

    void runOnOperation() final;

    bool checkUseVectorExtension(Operation* op) const {
        return op->hasTrait<mlir::OpTrait::VectorExtensionSupport>();
    }
};

void MarkVectorExtensionOpsPass::runOnOperation() {
    func::FuncOp f = getOperation();
    f->walk([&](Operation* op) {
        OpBuilder builder(op);
        if(checkUseVectorExtension(op)) {
            switch(cfg.vector_extension) {
                case VectorExtensions::AVX512:
                    op->setAttr("vector_extension", builder.getStringAttr("AVX512"));
                    break;
                case VectorExtensions::AVX2:
                    op->setAttr("vector_extension", builder.getStringAttr("AVX2"));
                    break;
                case VectorExtensions::SSE:
                    op->setAttr("vector_extension", builder.getStringAttr("SSE"));
                    break;
                case VectorExtensions::SCALAR:
                    op->setAttr("vector_extension", builder.getStringAttr("SCALAR"));
                    break;
            }
        }
        WalkResult::advance();
    });
}

std::unique_ptr<Pass> daphne::createMarkVectorExtensionOpsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<MarkVectorExtensionOpsPass>(cfg);
}
#endif 
