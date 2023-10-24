/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>

/**
 * @brief Inserts profiling tracepoints
 */
struct ProfilingPass: public PassWrapper<ProfilingPass, OperationPass<func::FuncOp>>
{
    explicit ProfilingPass() {}
    void runOnOperation() final;
};

void ProfilingPass::runOnOperation()
{
    func::FuncOp f = getOperation();
    Block & b = f.getBody().front();

    OpBuilder builder(&b, b.begin());
    Location loc = builder.getUnknownLoc();

    builder.create<daphne::StartProfilingOp>(loc);
    builder.setInsertionPoint(b.getTerminator());
    builder.create<daphne::StopProfilingOp>(loc);
}

std::unique_ptr<Pass> daphne::createProfilingPass()
{
    return std::make_unique<ProfilingPass>();
}
