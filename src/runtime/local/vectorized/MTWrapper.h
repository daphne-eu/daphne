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

#include <ir/daphneir/Daphne.h>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>
#include <runtime/local/vectorized/WorkerCPU.h>
#include <runtime/local/vectorized/WorkerGPU.h>

#include <spdlog/spdlog.h>

#include <fstream>
#include <functional>
#include <queue>
#include <set>

//TODO use the wrapper to cache threads
//TODO generalize for arbitrary inputs (not just binary)

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

template <typename DT>
class MTWrapperBase {
protected:
    std::vector<std::unique_ptr<Worker>> cuda_workers;
    std::vector<std::unique_ptr<Worker>> cpp_workers;
    std::vector<int> topologyPhysicalIds;
    std::vector<int> topologyUniqueThreads;
    std::vector<int> topologyResponsibleThreads;
    std::string _cpuinfoPath = "/proc/cpuinfo";
    size_t _numThreads{};
    uint32_t _numCPPThreads{};
    uint32_t _numCUDAThreads{};
    int _queueMode;
    // _queueMode 0: Centralized queue for all workers, 1: One queue for every physical ID (socket), 2: One queue per CPU
    int _numQueues;
    int _stealLogic;
    int _totalNumaDomains;
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

    bool _parseStringLine(const std::string& input, const std::string& keyword, int *val ) {
        auto seperatorLocation = input.find(':');
        if(seperatorLocation != std::string::npos) {
            if(input.find(keyword) == 0) {
                *val = stoi(input.substr(seperatorLocation + 1));
                return true;
            }
        }
        return false;
    }

    // FIXME: This method is not working on ARM - see GitHub issue [DAPHNE-#554]
    void get_topology(std::vector<int> &physicalIds, std::vector<int> &uniqueThreads, std::vector<int> &responsibleThreads) {
        std::ifstream cpuinfoFile(_cpuinfoPath);
        std::vector<int> utilizedThreads;
        std::vector<int> core_ids;
        int index = 0;
        if( cpuinfoFile.is_open() ) {
            std::string line;
            int value;
            while ( std::getline(cpuinfoFile, line) ) {
                if( _parseStringLine(line, "processor", &value ) ) {
                    utilizedThreads.push_back(value);
                } else if( _parseStringLine(line, "physical id", &value) ) {
                    if ( _ctx->getUserConfig().queueSetupScheme == PERGROUP ) {
                        if (std::find(physicalIds.begin(), physicalIds.end(), value) == physicalIds.end()) {
                            responsibleThreads.push_back(utilizedThreads[index]);
                        }
                    }
                    physicalIds.push_back(value);
                } else if( _parseStringLine(line, "core id", &value) ) {
                    int found = 0;
                    for (int i=0; i<index; i++) {
                        if (core_ids[i] == value && physicalIds[i] == physicalIds[index]) {
                                found++;
                        }
                    }
                    core_ids.push_back(value);
                    if( _ctx->config.hyperthreadingEnabled || found == 0 ) {
                        uniqueThreads.push_back(utilizedThreads[index]);
                        if ( _ctx->getUserConfig().queueSetupScheme == PERCPU ) {
                            responsibleThreads.push_back(value);
                        } else if ( _ctx->getUserConfig().queueSetupScheme == CENTRALIZED ) {
                            responsibleThreads.push_back(0);
                        }
                    }
                    index++;
                }
            }
            cpuinfoFile.close();
        }
    }
    void initCPPWorkers(std::vector<TaskQueue *> &qvector, uint32_t batchSize, const bool verbose = false,
            int numQueues = 0, int queueMode = 0, bool pinWorkers = false) {
        cpp_workers.resize(_numCPPThreads);
        if( numQueues == 0 ) {
            throw std::runtime_error("MTWrapper::initCPPWorkers: numQueues is 0, this should not happen.");
        }

        int i = 0;
        for( auto& w : cpp_workers ) {
            w = std::make_unique<WorkerCPU>(qvector, topologyPhysicalIds, topologyUniqueThreads, _ctx, verbose, 0, batchSize,
                    i, numQueues, queueMode, this->_stealLogic, pinWorkers);
            i++;
        }
    }
#ifdef USE_CUDA
    void initCUDAWorkers(TaskQueue* q, uint32_t batchSize, bool verbose = false) {
        cuda_workers.resize(_numCUDAThreads);
        for (auto& w : cuda_workers)
            w = std::make_unique<WorkerGPU>(q, _ctx, verbose, 1, batchSize);
    }

    void cudaPrefetchInputs(Structure** inputs, uint32_t numInputs, size_t mem_required,
            mlir::daphne::VectorSplit* splits) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(_ctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(_ctx, deviceID);
        auto buffer_usage = static_cast<float>(mem_required) / static_cast<float>(ctx->getMemBudget());
        ctx->logger->debug("Vect pipe total in/out buffer usage: {}", buffer_usage);
        if(buffer_usage < 1.0) {
            for (auto i = 0u; i < numInputs; ++i) {
                if(splits[i] == mlir::daphne::VectorSplit::ROWS) {
                    [[maybe_unused]] auto unused = static_cast<const DT*>(inputs[i])->getValues(&alloc_desc);
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
                mem_required += static_cast<DT*>((*res[i]))->getBufferSize();
            }
        }
        return mem_required;
    }

    virtual void combineOutputs(DT***& res, DT***& res_cuda, size_t numOutputs, mlir::daphne::VectorCombine* combines,
            DCTX(ctx)) = 0;

    void joinAll() {
        for(auto& w : cpp_workers)
            w->join();
        for(auto& w : cuda_workers)
            w->join();
    }

public:
    explicit MTWrapperBase(uint32_t numFunctions, DCTX(ctx)) : _ctx(ctx) {
        // ToDo: this is a workaround until getTopology() is properly fixed via hwloc library calls (see issue [DAPHNE-#554])
#ifdef __x86_64__
        _ctx->logger->debug("Querying x86-64 cpu topology");
        get_topology(topologyPhysicalIds, topologyUniqueThreads, topologyResponsibleThreads);
#else
        _ctx->logger->debug("Querying arm cpu topology");
        _numCPPThreads = std::thread::hardware_concurrency();
        for (auto i = 0; i < static_cast<int>(_numCPPThreads); i++) {
            topologyPhysicalIds.push_back(i);
            topologyUniqueThreads.push_back(i);
            topologyResponsibleThreads.push_back(i);
        }
#endif
        if(ctx->config.numberOfThreads > 0)
            _numCPPThreads = ctx->config.numberOfThreads;
        else
            _numCPPThreads = std::thread::hardware_concurrency();

        if(_ctx->getUserConfig().queueSetupScheme != CENTRALIZED)
            _numCPPThreads = topologyUniqueThreads.size();

        // If the available CPUs from Slurm is less than the configured num threads, use the value from Slurm
        if(const char* env_m = std::getenv("SLURM_CPUS_ON_NODE"))
            if(std::stoul(env_m) < _numCPPThreads)
                _numCPPThreads = std::stoi(env_m);

        // this is a bit hacky: if the second function (if available) is assumed to be the one containing CUDA ops
        if(ctx && ctx->useCUDA() && numFunctions > 1)
            _numCUDAThreads = ctx->cuda_contexts.size();

        _queueMode = 0;
        _numQueues = 1;
        _stealLogic = _ctx->getUserConfig().victimSelection;
        if( std::thread::hardware_concurrency() < topologyUniqueThreads.size() && _ctx->config.hyperthreadingEnabled )
            topologyUniqueThreads.resize(_numCPPThreads);
        _numThreads = _numCPPThreads + _numCUDAThreads;
        _totalNumaDomains = std::set<double>( topologyPhysicalIds.begin(), topologyPhysicalIds.end() ).size();

        if ( _ctx->getUserConfig().queueSetupScheme == PERGROUP ) {
            _queueMode = 1;
            _numQueues = _totalNumaDomains;
        } else if ( _ctx->getUserConfig().queueSetupScheme == PERCPU ) {
            _queueMode = 2;
            _numQueues = _numCPPThreads;
        }

        // ToDo: use logger
        if( _ctx->config.debugMultiThreading ) {
            std::cout << "topologyPhysicalIds:" << std::endl;
            for(const auto & topologyEntry: topologyPhysicalIds) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "topologyUniqueThreads:" << std::endl;
            for(const auto & topologyEntry: topologyUniqueThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "topologyResponsibleThreads:" << std::endl;
            for(const auto & topologyEntry: topologyResponsibleThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl << "_totalNumaDomains=" << _totalNumaDomains << std::endl;
            std::cout << "_numQueues=" << _numQueues << std::endl;
        }

        _ctx->logger->debug("spawning {} CPU and {} CUDA worker threads", this->_numCPPThreads, this->_numCUDAThreads);
    }

    virtual ~MTWrapperBase() = default;
};

template<typename DT>
class MTWrapper : public MTWrapperBase<DT> {};

template<typename VT>
class MTWrapper<DenseMatrix<VT>> : public  MTWrapperBase<DenseMatrix<VT>> {
public:
    using PipelineFunc = void(DenseMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numFunctions, DCTX(ctx)) : MTWrapperBase<DenseMatrix<VT>>(numFunctions, ctx){}


    [[maybe_unused]] void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res,
            const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
            int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res,
            const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t *outRows,
            int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res,
            const bool* isScalar,Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows,
            int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    void combineOutputs(DenseMatrix<VT>***& res, DenseMatrix<VT>***& res_cuda, size_t numOutputs,
            mlir::daphne::VectorCombine* combines, DCTX(ctx)) override;
};

template<typename VT>
class MTWrapper<CSRMatrix<VT>> : public MTWrapperBase<CSRMatrix<VT>> {
public:
    using PipelineFunc = void(CSRMatrix<VT> ***, Structure **, DCTX(ctx));

    explicit MTWrapper(uint32_t numFunctions, DCTX(ctx)) :
            MTWrapperBase<CSRMatrix<VT>>(numFunctions, ctx){ }

    [[maybe_unused]] void executeSingleQueue(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res,
            const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, const int64_t* outRows,
            const int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose) {
        throw std::runtime_error("sparse single queue vect exec not implemented");
    }

    [[maybe_unused]] void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res,
            const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, const int64_t* outRows,
            const int64_t* outCols, VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res,
            const bool* isScalar, Structure** inputs, size_t numInputs, size_t numOutputs, int64_t* outRows, int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose) {
        throw std::runtime_error("sparse queuePerDeviceType vect exec not implemented");
    }
    
    void combineOutputs(CSRMatrix<VT>***& res, CSRMatrix<VT>***& res_cuda, [[maybe_unused]] size_t numOutputs,
                        [[maybe_unused]] mlir::daphne::VectorCombine* combines, DCTX(ctx)) override {}
};