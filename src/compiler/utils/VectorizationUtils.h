/*
 *  Copyright 2025 The DAPHNE Consortium
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

#include <mlir/IR/Operation.h>

struct VectorizationUtils {
    /**
     * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
     *
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    static bool valueDependsOnResultOf(mlir::Value value, mlir::Operation *op);

    /**
     * @brief Recursive function checking if any operand of the given operation `src` is transitively dependant on the
     * operation `op`.
     *
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    static bool operationDependsOnResultOf(mlir::Operation *src, mlir::Operation *op);

    /**
     * @brief Moves operations which are between the operations to be fused into a single pipeline before or after the
     * position where the pipeline will be placed.
     *
     * @param pipelinePosition The position where the pipeline will be
     * @param pipeline The pipeline for which this function should be executed
     */
    static void movePipelineInterleavedOperations(mlir::Block::iterator pipelinePosition,
                                                  const std::vector<mlir::Operation *> &pipeline);
};