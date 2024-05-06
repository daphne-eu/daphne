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
#include <util/KernelDispatchMapping.h>

#include <mlir/Pass/Pass.h>

using namespace mlir;

#include <iostream>

/**
 * @brief Inserts a DaphneIR `CreateDaphneContextOp` and a
 * `DestroyDaphneContextOp` as the first and last (before the terminator)
 * operation in each function.
 */
// TODO For now, this should be fine, but in the future, we can think about
// extensions in several directions, e.g.:
// - inserting the context into blocks (e.g. parfor loop bodies)
// - passing the context as an argument to a function
struct InsertDaphneContextPass : public PassWrapper<InsertDaphneContextPass, OperationPass<func::FuncOp>>
{
    const DaphneUserConfig& user_config;
    explicit InsertDaphneContextPass(const DaphneUserConfig& cfg) : user_config(cfg) {}
    void runOnOperation() final;
};

void InsertDaphneContextPass::runOnOperation()
{
    func::FuncOp f = getOperation();
    Block & b = f.getBody().front();
    
    OpBuilder builder(&b, b.begin());
    Location loc = f.getLoc();

    // Insert a CreateDaphneContextOp as the first operation in the block.
    builder.create<daphne::CreateDaphneContextOp>(
        loc, daphne::DaphneContextType::get(&getContext()),
        builder.create<daphne::ConstantOp>(
            loc, reinterpret_cast<uint64_t>(&user_config)),
        builder.create<daphne::ConstantOp>(
            loc,
            reinterpret_cast<uint64_t>(&KernelDispatchMapping::instance())));
#ifdef USE_CUDA
    if(user_config.use_cuda) {
        builder.create<daphne::CreateCUDAContextOp>(loc);
    }
#endif
    if (user_config.use_distributed){
        builder.create<daphne::CreateDistributedContextOp>(loc);
    }
#ifdef USE_FPGAOPENCL
    if(user_config.use_fpgaopencl) {
        builder.create<daphne::CreateFPGAContextOp>(loc);
    }
#endif

 
    // Insert a DestroyDaphneContextOp as the last operation in the block, but
    // before the block's terminator.
    builder.setInsertionPoint(b.getTerminator());
    builder.create<daphne::DestroyDaphneContextOp>(loc);
}

std::unique_ptr<Pass> daphne::createInsertDaphneContextPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<InsertDaphneContextPass>(cfg);
}
