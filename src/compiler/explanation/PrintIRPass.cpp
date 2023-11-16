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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <iostream>
#include <string>

using namespace mlir;

/**
 * @brief A compiler pass that simply prints the IR.
 *
 * Useful for manual testing and debugging, since this pass can easily be
 * integrated after any other pass to have a look at the IR.
 */
class PrintIRPass : public PassWrapper<PrintIRPass, OperationPass<ModuleOp>> {
    std::string message;

   public:
    PrintIRPass(const std::string message) : message(message) {}

    void runOnOperation() final;

    StringRef getArgument() const final { return "print-ir"; }
    StringRef getDescription() const final {
        return "Pass for debugging purposes, prints the IR at the current "
               "stage in the compilation pipeline.";
    }
};

void PrintIRPass::runOnOperation() {
    std::cerr << message << std::endl;

    auto module = getOperation();
    OpPrintingFlags flags = {};
    flags.enableDebugInfo(/*enable=*/false, /*prettyForm=*/false);
    module.print(llvm::errs(), flags);
}

std::unique_ptr<Pass> daphne::createPrintIRPass(const std::string message) {
    return std::make_unique<PrintIRPass>(message);
}
