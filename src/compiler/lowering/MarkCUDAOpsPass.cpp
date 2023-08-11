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
#include "runtime/local/context/CUDAContext.h"
#include <mlir/IR/IRMapping.h>

using namespace mlir;

struct MarkCUDAOpsPass : public PassWrapper<MarkCUDAOpsPass, OperationPass<func::FuncOp>> {
    
    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;
    size_t available_gpu_mem{};
    size_t total_gpu_mem{};
    size_t mem_budget;
    std::shared_ptr<spdlog::logger> logger;

    explicit MarkCUDAOpsPass(const DaphneUserConfig& cfg) : cfg(cfg) {
        // ToDo: use context and per device mem info
        cudaMemGetInfo(&available_gpu_mem, &total_gpu_mem);
        mem_budget = std::floor(0.9 * static_cast<double>(total_gpu_mem));
        logger = spdlog::get("compiler::cuda");
    }
    
    void runOnOperation() final;
    
    void addCUDAOpsToVectorizedPipeline(OpBuilder& builder, daphne::VectorizedPipelineOp& pipelineOp) const {
        
        auto& pipeline = pipelineOp.getBody().front().getOperations();
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
                    logger->trace("checking pipeline op for cuda: {}: {}", o.getName().getStringRef().str(), out);
                    return !out;
                }
                else return false;
            });
            build_cuda_pipeline = !pipeline_has_unsupported_cuda_ops;
        }
        
        // clone body region into cuda region if there's a cuda supported op in body
        if(build_cuda_pipeline) {
            PatternRewriter::InsertionGuard insertGuard(builder);
            IRMapping mapper;
            pipelineOp.getBody().cloneInto(&pipelineOp.getCuda(), mapper);
            for (auto &op: pipelineOp.getCuda().front().getOperations()) {
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
                    logger->warn("Ignoring unknown dimension in max mem check of {}"
                            "dims are: {}x{}\nsetting unknowns to 1 for this test", op->getName().getStringRef().str(),
                            rows, cols);
                    if(rows < 0)
                        rows = 1;
                    if(cols < 0)
                        cols = 1;
                }
                opSize += rows * cols * t.getElementType().getIntOrFloatBitWidth() / 8;
            }
        }
        auto inSize = opSize;
        logger->trace("op in size: {} kb", opSize / 1024);
        for(auto result : op->getResults()) {
            auto type = result.getType();
            if(auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
                opSize += t.getNumRows() * t.getNumCols() * t.getElementType().getIntOrFloatBitWidth() / 8;
            }
        }
        logger->debug("op out size: {} kb\ntotal op size: {} mb", (opSize-inSize) / 1024,
                opSize / 1048576);

        if(opSize < mem_budget)
            return true;
        else
            return false;
    }
    
    // ToDo: requirements should be set per operator in tablegen
    bool hasReqMinDims(mlir::Operation* op) const {
        auto checkDims = [this,op](const mlir::Type& type) -> bool {
            if(auto t = type.dyn_cast<mlir::daphne::MatrixType>()) {
                auto rows = t.getNumRows();
                auto cols = t.getNumCols();
                if(rows < 0 || cols < 0) {
                    logger->warn("Ignoring unknown dimension in min input size check of {} dims are: {}x{}\nsetting "
                            "unknowns to 256 for this test", op->getName().getStringRef().str(), rows, cols);
                    if(rows < 0)
                        rows = 256;
                    if(cols < 0)
                        cols = 256;
                }
                return (rows > 255 || cols > 255);
            }
            return false;
        };

        bool ret = false;
        for(auto type : op->getOperandTypes()) {
            if((ret = checkDims(type)))
                break;
        }

        if(!ret) {
            for (auto type: op->getResultTypes()) {
                if((ret = checkDims(type)))
                    break;
            }
        }
        return ret;
    }
    
    bool checkUseCUDA(Operation* op) const {
        logger->trace("checkUseCUDA: {}", op->getName().getStringRef().str());
        bool use_cuda = op->hasTrait<mlir::OpTrait::CUDASupport>();
        logger->trace("{} CUDA supported={}", op->getName().getStringRef().str(), use_cuda);
        use_cuda = use_cuda && CompilerUtils::isMatrixComputation(op);
        logger->trace("{} isMatrixComputation={}", op->getName().getStringRef().str(), use_cuda);
        use_cuda = use_cuda && hasReqMinDims(op);
        logger->trace("{} hasMinInputDims={}", op->getName().getStringRef().str(), use_cuda);
        use_cuda = use_cuda && fitsInMemory(op);
        logger->trace("{} fitsInMem={}", op->getName().getStringRef().str(), use_cuda);
        return use_cuda;
    }
};

void MarkCUDAOpsPass::runOnOperation() {
    getOperation()->walk([&](Operation* op) {
        logger->debug("MarkCUDAOpsPass: {} parent: {}", op->getName().getStringRef().str(),
                op->getParentOp()->getName().getStringRef().str());
        OpBuilder builder(op);
        // handle vectorizedPipelineOps
        if (auto constOp = llvm::dyn_cast<daphne::ConstantOp>(op))
        {
            WalkResult::advance();
            return;
        }
        else if (auto pipelineOp = llvm::dyn_cast<daphne::VectorizedPipelineOp>(op))
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

std::unique_ptr<Pass> daphne::createMarkCUDAOpsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<MarkCUDAOpsPass>(cfg);
}

#endif // USE_CUDA