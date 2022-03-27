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

#pragma once

#include <runtime/local/vectorized/TaskQueues.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>
#include <runtime/local/vectorized/Workers.h>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <ir/daphneir/Daphne.h>

#include <functional>
#include <queue>

//TODO use the wrapper to cache threads
//TODO generalize for arbitrary inputs (not just binary)

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

template <typename DT>
class MTWrapperBase {
protected:
    std::vector<std::unique_ptr<Worker>> cuda_workers;
    std::vector<std::unique_ptr<Worker>> cpp_workers;
    uint32_t _numThreads{};
    uint32_t _numCPPThreads{};
    uint32_t _numCUDAThreads{};
    DCTX(_ctx);

    std::pair<size_t, size_t> getInputProperties(Structure** inputs, size_t numInputs, VectorSplit* splits) {
        auto len = 0ul;
        auto mem_required = 0ul;

        // due to possible broadcasting we have to check all inputs
        for (auto i = 0u; i < numInputs; ++i) {
            if (splits[i] == mlir::daphne::VectorSplit::ROWS) {
                len = std::max(len, inputs[i]->getNumRows());
                mem_required += inputs[i]->getNumItems() * sizeof(typename DT::VT);
            }
        }
        return std::make_pair(len, mem_required);
    }

    void initCPPWorkers(TaskQueue* q, uint32_t batchSize, bool verbose = false) {
        cpp_workers.resize(_numCPPThreads);
        for(auto& w : cpp_workers)
            w = std::make_unique<WorkerCPU>(q, verbose, 0, batchSize);
    }

#ifdef USE_CUDA
    void initCUDAWorkers(TaskQueue* q, uint32_t batchSize, bool verbose = false) {
        cuda_workers.resize(_numCUDAThreads);
        for (auto& w : cuda_workers)
            w = std::make_unique<WorkerCPU>(q, verbose, 1, batchSize);
    }

    void cudaPrefetchInputs(Structure** inputs, uint32_t numInputs, size_t mem_required,
            mlir::daphne::VectorSplit* splits) {
        // ToDo: multi-device support :-P
        auto cctx = _ctx->getCUDAContext(0);
        auto buffer_usage = static_cast<float>(mem_required) / static_cast<float>(cctx->getMemBudget());
#ifndef NDEBUG
        std::cout << "\nVect pipe total in/out buffer usage: " << buffer_usage << std::endl;
#endif
        if(buffer_usage < 1.0) {
            for (auto i = 0u; i < numInputs; ++i) {
                if(splits[i] == mlir::daphne::VectorSplit::ROWS) {
                    [[maybe_unused]] auto bla = static_cast<const DT*>(inputs[i])->getValuesCUDA();
                }
            }
        }
    }
#endif
    size_t allocateOutput(DT***& res, size_t numOutputs, const int64_t* outRows, const int64_t* outCols,
            mlir::daphne::VectorCombine* combines) {
        auto mem_required = 0ul;
        // output allocation for row-wise combine
        for(size_t i = 0; i < numOutputs; ++i) {
            if((*res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                (*res[i]) = DataObjectFactory::create<DT>(outRows[i], outCols[i], zeroOut);
                mem_required += static_cast<DT*>((*res[i]))->bufferSize();
            }
        }
        return mem_required;
    }

    virtual void combineOutputs(DT***& res, DT***& res_cuda, size_t numOutputs, mlir::daphne::VectorCombine* combines) = 0;

    void joinAll() {
        for(auto& w : cpp_workers)
            w->join();
        for(auto& w : cuda_workers)
            w->join();
    }

public:
    explicit MTWrapperBase(uint32_t numThreads, uint32_t numFunctions, DCTX(ctx)) : _ctx(ctx) {
        if(ctx->config.numberOfThreads > 0){
            _numThreads = ctx->config.numberOfThreads;
        }
        else{
            _numThreads = std::thread::hardware_concurrency();
        }
        if(ctx && ctx->useCUDA() && numFunctions > 1)
            _numCUDAThreads = ctx->cuda_contexts.size();
        _numCPPThreads = _numThreads;
        _numThreads = _numCPPThreads + _numCUDAThreads;
#ifndef NDEBUG
        std::cout << "spawning " << this->_numCPPThreads << " CPU and " << this->_numCUDAThreads << " CUDA worker threads"
                  << std::endl;
#endif
    }

    virtual ~MTWrapperBase() = default;
};

template<typename DT>
class MTWrapper : public MTWrapperBase<DT> {};

template<typename VT>
class MTWrapper<DenseMatrix<VT>> : public  MTWrapperBase<DenseMatrix<VT>> {
public:
    using PipelineFunc = void(DenseMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numThreads, uint32_t numFunctions, DCTX(ctx)) :
            MTWrapperBase<DenseMatrix<VT>>(numThreads, numFunctions, ctx){}

    void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res, bool* isScalar, Structure** inputs,
            size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t* outCols, VectorSplit* splits,
            VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res, bool* isScalar,
            Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows, int64_t* outCols,
            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    void combineOutputs(DenseMatrix<VT>***& res, DenseMatrix<VT>***& res_cuda, size_t numOutputs,
            mlir::daphne::VectorCombine* combines) override;
};

template<typename VT>
class MTWrapper<CSRMatrix<VT>> : public MTWrapperBase<CSRMatrix<VT>> {
public:
    using PipelineFunc = void(CSRMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numThreads, uint32_t numFunctions, DCTX(ctx)) :
            MTWrapperBase<CSRMatrix<VT>>(numThreads, numFunctions, ctx){}

    void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res, bool* isScalar, Structure** inputs,
                            size_t numInputs, size_t numOutputs, const int64_t* outRows, const int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res, bool* isScalar, Structure** inputs,
                            size_t numInputs, size_t numOutputs, int64_t* outRows, int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    void combineOutputs(CSRMatrix<VT>***& res, CSRMatrix<VT>***& res_cuda, [[maybe_unused]] size_t numOutputs,
                        [[maybe_unused]] mlir::daphne::VectorCombine* combines) override {}
};