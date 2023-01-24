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
#ifdef USE_CUDA
#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
//#include "runtime/local/context/CUDAContext.h"
#include <mlir/IR/BlockAndValueMapping.h>

#include <iostream>

using namespace mlir;

struct MarkCUDAOpsPass : public PassWrapper<MarkCUDAOpsPass, FunctionPass> {
    
    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;
    size_t available_mem{};
    size_t total_mem{};
    size_t mem_budget;
    
    explicit MarkCUDAOpsPass(const DaphneUserConfig& cfg, const size_t& available_gpu_mem, const size_t& total_gpu_mem)
            : cfg(cfg), available_mem(available_gpu_mem), total_mem(total_gpu_mem) {
        // ToDo: use context and per device mem info
        //cudaMemGetInfo(&available_gpu_mem, &total_gpu_mem);
//        available_gpu_mem = 1000000*256;
//        total_gpu_mem = 1000000*512;
//        std::cout << "querying mem info" << std::endl;
//        cuda_get_mem_info(&available_gpu_mem, &total_gpu_mem);
//        std::cout << "availabpe_gpu_mem: " << available_gpu_mem << std::endl;
        mem_budget = std::floor(0.9 * static_cast<double>(total_mem));
        if(mem_budget > available_mem)
            mem_budget = std::floor(0.9 * static_cast<double>(available_mem));
    }
    
    void runOnFunction() final;
    
    void addCUDAOpsToVectorizedPipeline(OpBuilder& builder, daphne::VectorizedPipelineOp& pipelineOp) const {
        
        auto& pipeline = pipelineOp.body().front().getOperations();
        bool build_cuda_pipeline;
        
        // add CUDA ops if at least one (cuda_fuse_any) or all (!cuda_fuse_any) ops would be supported
        if(cfg.cuda_fuse_any) {
            bool pipeline_has_supported_cuda_ops = llvm::any_of(pipeline, [&](Operation& o) {
                return llvm::isa<daphne::ReturnOp>(o) || checkUseCUDA(&o);
            });
            build_cuda_pipeline = pipeline_has_supported_cuda_ops;
        }
        else {
            bool pipeline_has_unsupported_cuda_ops = llvm::any_of(pipeline, [&](Operation& o) {
                if (!llvm::isa<daphne::ReturnOp>(o)) {
                    bool out = checkUseCUDA(&o);
//                    std::cout << "checking pipeline op for cuda: " << o.getName().getStringRef().str() << ": " << out << std::endl;
                    return !out;
                }
                else return false;
            });
            build_cuda_pipeline = !pipeline_has_unsupported_cuda_ops;
        }
        
        // clone body region into cuda region if there's a cuda supported op in body
        if(build_cuda_pipeline) {
            PatternRewriter::InsertionGuard insertGuard(builder);
            BlockAndValueMapping mapper;
            pipelineOp.body().cloneInto(&pipelineOp.cuda(), mapper);
            for (auto &op: pipelineOp.cuda().front().getOperations()) {
                bool isMat = CompilerUtils::isMatrixComputation(&op);
                if (op.hasTrait<mlir::OpTrait::CUDASupport>() && isMat)
                    op.setAttr("cuda_device", builder.getI32IntegerAttr(0));
            }
        }
    }
    
    bool fitsInMemory(mlir::Operation* op) const {
        auto opSize = 0ul;
        for(auto operand : op->getOperands()) {
            auto type = operand.getType();
            if(auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
                auto rows = t.getNumRows();
                auto cols = t.getNumCols();
                if(rows < 0 || cols < 0) {
                    std::cerr << "Warning: Ignoring unknown dimension in max mem check of " <<
                              op->getName().getStringRef().str() << "\ndims are: " << rows << "x" << cols <<
                              "\nsetting unknowns to 1 for this test" << std::endl;
                    if(rows < 0)
                        rows = 1;
                    if(cols < 0)
                        cols = 1;
                }
                opSize += rows * cols * t.getElementType().getIntOrFloatBitWidth();
            }
        }
//        auto inSize = opSize;
//        std::cout << "op in size: " << opSize / 1024 << " kB" << std::endl;
        for(auto result : op->getResults()) {
            auto type = result.getType();
            if(auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
                opSize += t.getNumRows() * t.getNumCols() * t.getElementType().getIntOrFloatBitWidth();
            }
        }
//        std::cout << "op out size: " << (opSize-inSize) / 1024 << " kB" << std::endl;
        if(opSize < mem_budget)
            return true;
        else
            return false;
    }
    
    // ToDo: requirements should be set per operator in tablegen
    static bool hasReqMinDims(mlir::Operation* op) {
        for(auto operand : op->getOperands()) {
            auto type = operand.getType();
            if(auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
                auto rows = t.getNumRows();
                auto cols = t.getNumCols();
                if(rows < 0 || cols < 0) {
                    std::cerr << "Warning: Ignoring unknown dimension in min input size check of " <<
                              op->getName().getStringRef().str() << "\ndims are: " << rows << "x" << cols <<
                              "\nsetting unknowns to 256 for this test" << std::endl;
                    if(rows < 0)
                        rows = 256;
                    if(cols < 0)
                        cols = 256;
                }
                return (rows > 255 || cols > 255);
            }
        }
        return false;
    }
    
    bool checkUseCUDA(Operation* op) const {
//        std::cout << "checkUseCUDA: " << op->getName().getStringRef().str() << std::endl;
        
        bool use_cuda = op->hasTrait<mlir::OpTrait::CUDASupport>();
        use_cuda = use_cuda && CompilerUtils::isMatrixComputation(op);
        use_cuda = use_cuda && hasReqMinDims(op);
        use_cuda = use_cuda && fitsInMemory(op);
        return use_cuda;
    }
};

void MarkCUDAOpsPass::runOnFunction() {
    getFunction()->walk([&](Operation* op) {
//        std::cout << "MarkCUDAOpsPass: " << op->getName().getStringRef().str() << " parent: " << op->getParentOp()->getName().getStringRef().str() <<  std::endl;
        
        OpBuilder builder(op);
        // handle vectorizedPipelineOps
        if (auto pipelineOp = llvm::dyn_cast<daphne::VectorizedPipelineOp>(op))
            addCUDAOpsToVectorizedPipeline(builder, pipelineOp);
        else {
            if((!llvm::isa<daphne::VectorizedPipelineOp>(op->getParentOp()) && checkUseCUDA(op)) ||
                 llvm::isa<daphne::CreateCUDAContextOp>(op)) {
                op->setAttr("cuda_device", builder.getI32IntegerAttr(0));
            }
        }
        WalkResult::advance();
    });
}

std::unique_ptr<Pass> daphne::createMarkCUDAOpsPass(const DaphneUserConfig& cfg, const size_t& available_gpu_mem,
        const size_t& total_gpu_mem) {
    return std::make_unique<MarkCUDAOpsPass>(cfg, available_gpu_mem, total_gpu_mem);
}
#endif // USE_CUDA