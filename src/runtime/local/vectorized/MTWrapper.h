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

#ifdef USE_CUDA
#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"
#endif

#include "PipelineHWlocInfo.h"

#include <ir/daphneir/Daphne.h>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>
#include <runtime/local/vectorized/WorkerCPU.h>
#include <runtime/local/vectorized/WorkerGPU.h>

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

#include <hwloc.h>

// TODO generalize for arbitrary inputs (not just binary)

using mlir::daphne::VectorCombine;
using mlir::daphne::VectorSplit;

template <typename DT> class MTWrapperBase {
  protected:
    std::vector<std::unique_ptr<Worker>> cuda_workers;
    std::vector<std::unique_ptr<Worker>> cpp_workers;
    size_t _numThreads{};
    uint32_t _numCPPThreads{};
    uint32_t _numCUDAThreads{};
    QueueTypeOption _queueMode;
    int _numQueues;
    VictimSelectionLogic _victimSelection;
    int _totalNumaDomains;
    PipelineHWlocInfo _topology;
    DCTX(_ctx);

    std::pair<size_t, size_t> getInputProperties(Structure **inputs, size_t numInputs, VectorSplit *splits) {
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

    void initCPPWorkers(std::vector<TaskQueue *> &qvector, uint32_t batchSize, const bool verbose = false,
                        int numQueues = 0, QueueTypeOption queueMode = QueueTypeOption::CENTRALIZED,
                        bool pinWorkers = false) {
        cpp_workers.resize(_numCPPThreads);
        if (numQueues == 0) {
            throw std::runtime_error("MTWrapper::initCPPWorkers: numQueues is "
                                     "0, this should not happen.");
        }

        int i = 0;
        for (auto &w : cpp_workers) {
            _ctx->logger->debug("creating worker {} with topology {}, size={}", i, _topology.physicalIds,
                                _topology.physicalIds.size());
            w = std::make_unique<WorkerCPU>(qvector, _topology.physicalIds, _topology.uniqueThreads, _ctx, verbose, 0,
                                            batchSize, i, numQueues, queueMode, this->_victimSelection, pinWorkers);
            i++;
        }
    }
#ifdef USE_CUDA
    void initCUDAWorkers(TaskQueue *q, uint32_t batchSize, bool verbose = false) {
        cuda_workers.resize(_numCUDAThreads);
        for (auto &w : cuda_workers)
            w = std::make_unique<WorkerGPU>(q, _ctx, verbose, 1, batchSize);
    }

    void cudaPrefetchInputs(Structure **inputs, uint32_t numInputs, size_t mem_required,
                            mlir::daphne::VectorSplit *splits) {
        const size_t deviceID = 0; // ToDo: multi device support
        auto ctx = CUDAContext::get(_ctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(_ctx, deviceID);
        auto buffer_usage = static_cast<float>(mem_required) / static_cast<float>(ctx->getMemBudget());
        ctx->logger->debug("Vect pipe total in/out buffer usage: {}", buffer_usage);
        if (buffer_usage < 1.0) {
            for (auto i = 0u; i < numInputs; ++i) {
                if (splits[i] == mlir::daphne::VectorSplit::ROWS) {
                    [[maybe_unused]] auto unused = static_cast<const DT *>(inputs[i])->getValues(&alloc_desc);
                }
            }
        }
    }
#endif
    size_t allocateOutput(DT ***&res, size_t numOutputs, const int64_t *outRows, const int64_t *outCols,
                          mlir::daphne::VectorCombine *combines) {
        auto mem_required = 0ul;
        // Output allocation and, in case of aggregating combine, also initialization.
        for (size_t i = 0; i < numOutputs; ++i) {
            if ((*res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                // TODO Ideally, we should not initialize the result for aggregating combines (since it wastes cycles),
                // but rather start the aggregation with the first individual result.
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                (*res[i]) = DataObjectFactory::create<DT>(outRows[i], outCols[i], zeroOut);
                if (combines[i] == mlir::daphne::VectorCombine::MIN)
                    std::fill((*res[i])->getValues(), (*res[i])->getValues() + outRows[i] * outCols[i],
                              std::is_floating_point_v<typename DT::VT>
                                  ? std::numeric_limits<typename DT::VT>::infinity()
                                  : std::numeric_limits<typename DT::VT>::max());
                else if (combines[i] == mlir::daphne::VectorCombine::MAX)
                    std::fill((*res[i])->getValues(), (*res[i])->getValues() + outRows[i] * outCols[i],
                              std::is_floating_point_v<typename DT::VT>
                                  ? -std::numeric_limits<typename DT::VT>::infinity()
                                  : std::numeric_limits<typename DT::VT>::min());
                mem_required += static_cast<DT *>((*res[i]))->getBufferSize();
            }
        }
        return mem_required;
    }

    virtual void combineOutputs(DT ***&res, DT ***&res_cuda, size_t numOutputs, mlir::daphne::VectorCombine *combines,
                                DCTX(ctx)) = 0;

    void joinAll() {
        for (auto &w : cpp_workers)
            w->join();
        for (auto &w : cuda_workers)
            w->join();
    }

  public:
    explicit MTWrapperBase(uint32_t numFunctions, PipelineHWlocInfo topology, DCTX(ctx))
        : _topology(std::move(topology)), _ctx(ctx) {
        _ctx->logger->debug("Querying cpu topology");

        if (ctx->config.numberOfThreads > 0)
            _numCPPThreads = ctx->config.numberOfThreads;
        else
            _numCPPThreads = _topology.physicalIds.size();

        if (_ctx->getUserConfig().queueSetupScheme != QueueTypeOption::CENTRALIZED)
            _numCPPThreads = _topology.uniqueThreads.size();

        // If the available CPUs from Slurm is less than the configured num
        // threads, use the value from Slurm
        if (const char *env_m = std::getenv("SLURM_CPUS_ON_NODE"))
            if (std::stoul(env_m) < _numCPPThreads)
                _numCPPThreads = std::stoi(env_m);

        // this is a bit hacky: if the second function (if available) is assumed
        // to be the one containing CUDA ops
        if (ctx && ctx->useCUDA() && numFunctions > 1)
            _numCUDAThreads = ctx->cuda_contexts.size();

        _queueMode = QueueTypeOption::CENTRALIZED;
        _numQueues = 1;
        _victimSelection = _ctx->getUserConfig().victimSelection;
        if (std::thread::hardware_concurrency() < _topology.uniqueThreads.size() && _ctx->config.hyperthreadingEnabled)
            _topology.uniqueThreads.resize(_numCPPThreads);
        _numThreads = _numCPPThreads + _numCUDAThreads;
        _totalNumaDomains = std::set<double>(_topology.physicalIds.begin(), _topology.physicalIds.end()).size();

        if (_ctx->getUserConfig().queueSetupScheme == QueueTypeOption::PERGROUP) {
            _queueMode = QueueTypeOption::PERGROUP;
            _numQueues = _totalNumaDomains;
        } else if (_ctx->getUserConfig().queueSetupScheme == QueueTypeOption::PERCPU) {
            _queueMode = QueueTypeOption::PERCPU;
            _numQueues = _numCPPThreads;
        }

        // ToDo: use logger
        if (_ctx->config.debugMultiThreading) {
            std::cout << "topologyPhysicalIds:" << std::endl;
            for (const auto &topologyEntry : _topology.physicalIds) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "topologyUniqueThreads:" << std::endl;
            for (const auto &topologyEntry : _topology.uniqueThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "topologyResponsibleThreads:" << std::endl;
            for (const auto &topologyEntry : _topology.responsibleThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "_totalNumaDomains=" << _totalNumaDomains << std::endl;
            std::cout << "_numQueues=" << _numQueues << std::endl;
        }

        _ctx->logger->debug("spawning {} CPU and {} CUDA worker threads", this->_numCPPThreads, this->_numCUDAThreads);
    }

    virtual ~MTWrapperBase() = default;
};

template <typename DT> class MTWrapper : public MTWrapperBase<DT> {};

template <typename VT> class MTWrapper<DenseMatrix<VT>> : public MTWrapperBase<DenseMatrix<VT>> {
  public:
    using PipelineFunc = void(DenseMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numFunctions, PipelineHWlocInfo topology, DCTX(ctx))
        : MTWrapperBase<DenseMatrix<VT>>(numFunctions, topology, ctx) {}

    [[maybe_unused]] void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT> ***res,
                                             const bool *isScalar, Structure **inputs, size_t numInputs,
                                             size_t numOutputs, int64_t *outRows, int64_t *outCols, VectorSplit *splits,
                                             VectorCombine *combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT> ***res,
                                           const bool *isScalar, Structure **inputs, size_t numInputs,
                                           size_t numOutputs, int64_t *outRows, int64_t *outCols, VectorSplit *splits,
                                           VectorCombine *combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs,
                                                    DenseMatrix<VT> ***res, const bool *isScalar, Structure **inputs,
                                                    size_t numInputs, size_t numOutputs, int64_t *outRows,
                                                    int64_t *outCols, VectorSplit *splits, VectorCombine *combines,
                                                    DCTX(ctx), bool verbose);

    void combineOutputs(DenseMatrix<VT> ***&res, DenseMatrix<VT> ***&res_cuda, size_t numOutputs,
                        mlir::daphne::VectorCombine *combines, DCTX(ctx)) override;
};

template <typename VT> class MTWrapper<CSRMatrix<VT>> : public MTWrapperBase<CSRMatrix<VT>> {
  public:
    using PipelineFunc = void(CSRMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numFunctions, PipelineHWlocInfo topology, DCTX(ctx))
        : MTWrapperBase<CSRMatrix<VT>>(numFunctions, topology, ctx) {}

    [[maybe_unused]] void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT> ***res,
                                             const bool *isScalar, Structure **inputs, size_t numInputs,
                                             size_t numOutputs, const int64_t *outRows, const int64_t *outCols,
                                             VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
        throw std::runtime_error("sparse single queue vect exec not implemented");
    }

    [[maybe_unused]] void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT> ***res,
                                           const bool *isScalar, Structure **inputs, size_t numInputs,
                                           size_t numOutputs, const int64_t *outRows, const int64_t *outCols,
                                           VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs,
                                                    CSRMatrix<VT> ***res, const bool *isScalar, Structure **inputs,
                                                    size_t numInputs, size_t numOutputs, int64_t *outRows,
                                                    int64_t *outCols, VectorSplit *splits, VectorCombine *combines,
                                                    DCTX(ctx), bool verbose) {
        throw std::runtime_error("sparse queuePerDeviceType vect exec not implemented");
    }

    void combineOutputs(CSRMatrix<VT> ***&res, CSRMatrix<VT> ***&res_cuda, [[maybe_unused]] size_t numOutputs,
                        [[maybe_unused]] mlir::daphne::VectorCombine *combines, DCTX(ctx)) override {}
};
