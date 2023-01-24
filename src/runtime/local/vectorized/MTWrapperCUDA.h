/*
 * Copyright 2022 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/vectorized/IVectorizedExecutor.h>
#include "MTWrapper.h"
#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"

class CUDAVectorizedExecutor : public IVectorizedExecutor {
public:
    using PipelineFunc = typename IVectorizedExecutor::PipelineFunc;

//    void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, void*** res,
    void executeQueuePerDeviceType(int bla, void*** res,
            const bool* isScalar,Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows,
            int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose) override {
        std::cout << "executeQueuePerDeviceType" << std::endl;
    };
};

namespace CUDA {
    template<typename DT>
    class MTWrapperBase : public ::MTWrapperBase<DT> {
    public:
        using PipelineFunc = typename ::MTWrapperBase<DT>::PipelineFunc;

    explicit MTWrapperBase(uint32_t numFunctions, DCTX(ctx)) : ::MTWrapperBase<DT>(numFunctions, ctx) { }

    protected:
        void initCUDAWorkers(TaskQueue* q, uint32_t batchSize, bool verbose = false) {
            this->cuda_workers.resize(this->_numCUDAThreads);
            for (auto& w : this->cuda_workers)
                w = std::make_unique<WorkerGPU>(q, verbose, 1, batchSize);
        }

        void cudaPrefetchInputs(Structure** inputs, uint32_t numInputs, size_t mem_required,
                                mlir::daphne::VectorSplit* splits) {
            const size_t deviceID = 0; //ToDo: multi device support
            auto ctx = CUDAContext::get(this->_ctx, deviceID);
            AllocationDescriptorCUDA alloc_desc(this->_ctx, deviceID);
            auto buffer_usage = static_cast<float>(mem_required) / static_cast<float>(ctx->getMemBudget());
#ifndef NDEBUG
            std::cout << "\nVect pipe total in/out buffer usage: " << buffer_usage << std::endl;
#endif
            if(buffer_usage < 1.0) {
                for (auto i = 0u; i < numInputs; ++i) {
                    if(splits[i] == mlir::daphne::VectorSplit::ROWS) {
                        [[maybe_unused]] auto unused = static_cast<const DT*>(inputs[i])->getValues(&alloc_desc);
                    }
                }
            }
        }

//        void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, DT*** res,
//                                const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
//                                int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    };

    template<typename DT>
    class MTWrapper : public MTWrapperBase<DT> {};

    template<typename VT>
    class MTWrapper<DenseMatrix<VT>> : public MTWrapperBase<DenseMatrix<VT>> {
    public:
        using PipelineFunc = typename ::MTWrapperBase<DenseMatrix<VT>>::PipelineFunc;
//        using ::MTWrapper<DenseMatrix<VT>>::getInputProperties;
//        using MTWrapperBase<DenseMatrix<VT>>::allocateOutput;
//        using MTWrapperBase<DenseMatrix<VT>>::initCPPWorkers;
//        using MTWrapper<DenseMatrix<VT>>::initCUDAWorkers;
//        using MTWrapper<DenseMatrix<VT>>::initCUDAWorkers;
        explicit MTWrapper(uint32_t numFunctions, DCTX(ctx)) : MTWrapperBase<DenseMatrix<VT>>(numFunctions, ctx) { }

        void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res,
                                        const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
                                        int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

        void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res,
                                               const bool* isScalar,Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows,
                                               int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

        void combineOutputs(DenseMatrix<VT>***& res, DenseMatrix<VT>***& res_cuda, size_t numOutputs,
                            mlir::daphne::VectorCombine* combines, DCTX(ctx));
    };

    template<typename VT>
    class MTWrapper<CSRMatrix<VT>> : public MTWrapperBase<CSRMatrix<VT>> {
    public:
        using PipelineFunc = typename ::MTWrapperBase<CSRMatrix<VT>>::PipelineFunc;
//        using MTWrapperBase<CSRMatrix<VT>>::getInputProperties;
//        using MTWrapperBase<CSRMatrix<VT>>::allocateOutput;
//        using MTWrapperBase<CSRMatrix<VT>>::initCPPWorkers;
//        using MTWrapperBase<CSRMatrix<VT>>::initCUDAWorkers;
//        using MTWrapper<CSRMatrix<VT>>::initCUDAWorkers;

        explicit MTWrapper(uint32_t numFunctions, DCTX(ctx)) : MTWrapperBase<CSRMatrix<VT>>(numFunctions, ctx) { }

        void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res,
                                const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
                                int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose){};

        void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res,
                                               const bool* isScalar,Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows,
                                               int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose){};

        void combineOutputs(CSRMatrix<VT>***& res, CSRMatrix<VT>***& res_cuda, size_t numOutputs,
                            mlir::daphne::VectorCombine* combines, DCTX(ctx)){};
    };
}