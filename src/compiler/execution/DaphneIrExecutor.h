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

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <api/cli/DaphneUserConfig.h>

class DaphneIrExecutor
{
public:
    DaphneIrExecutor(bool selectMatrixRepresentations, DaphneUserConfig cfg);

    bool runPasses(mlir::ModuleOp module);
    std::unique_ptr<mlir::ExecutionEngine> createExecutionEngine(mlir::ModuleOp module);

    mlir::MLIRContext *getContext()
    { return &context_; }
private:
    mlir::MLIRContext context_;
    DaphneUserConfig userConfig_;
    bool selectMatrixRepresentations_;
    // Storage for lib paths needed for StringRefs
    std::vector<std::string> sharedLibRefPaths;
};

