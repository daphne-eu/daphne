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
#include "mlir/Pass/PassManager.h"

#include <unordered_map>

class DaphneIrExecutor
{
public:
    DaphneIrExecutor(bool selectMatrixRepresentations, DaphneUserConfig cfg);

    bool runPasses(mlir::ModuleOp module);
    std::unique_ptr<mlir::ExecutionEngine> createExecutionEngine(mlir::ModuleOp module);

    mlir::MLIRContext *getContext()
    { return &context_; }

    DaphneUserConfig & getUserConfig() {
        return userConfig_;
    }

    const DaphneUserConfig & getUserConfig() const {
        return userConfig_;
    }

private:
    mlir::MLIRContext context_;
    DaphneUserConfig userConfig_;
    bool selectMatrixRepresentations_;
    // Storage for lib paths needed for StringRefs
    std::vector<std::string> sharedLibRefPaths;

    /**
     * @brief A map indicating which of the distinct kernels libraries known to the
     * kernel catalog are actually used in the MLIR module.
     *
     * This map gets pre-populated with `false` for each distinct library. The values
     * are set to `true` when a call to a pre-compiled kernel from that library is
     * created by this pass. This approach is thread-safe, since the structure of the
     * map does not change anymore. Thus, it can be used by multiple concurrent
     * instances of this pass.
     */
    std::unordered_map<std::string, bool> usedLibPaths;

    void buildCodegenPipeline(mlir::PassManager &);
};

