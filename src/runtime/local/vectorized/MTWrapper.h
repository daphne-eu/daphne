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
#include <fstream>
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
    std::string _cpuinfoPath = "/proc/cpuinfo";
    uint32_t _numThreads{};
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
    
    int _parseStringLine(const std::string& input, const std::string& keyword, int *val ) {
        std::size_t seperatorLocation = input.find(":");
        if (seperatorLocation!=std::string::npos) {
            if (input.find(keyword) == 0) {
                *val = stoi(input.substr(seperatorLocation+1));
                return 1;
            }
            return 0;
        }
        return 0;
    }

    void get_topology(std::vector<int> &physicalIds, std::vector<int> &uniqueThreads) {
        std::ifstream cpuinfoFile(_cpuinfoPath);
        std::vector<int> utilizedThreads;
        std::vector<int> core_ids;
        int index = 0;
        if( cpuinfoFile.is_open() ) {
            std::string line;
            int value;
            while (std::getline(cpuinfoFile, line)) {
                if( _parseStringLine(line, "processor", &value ) ) {
                    utilizedThreads.push_back(value);
                } else if( _parseStringLine(line, "physical id", &value) ) {
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
                    }
                    index++;
                }
            }
            cpuinfoFile.close();
        }
    }

    void initCPPWorkers(TaskQueue* q, uint32_t batchSize, bool verbose = false) {
        cpp_workers.resize(_numCPPThreads);
        for(auto& w : cpp_workers)
            w = std::make_unique<WorkerCPU>(q, verbose, 0, batchSize);
    }
    
    void initCPPWorkers(std::vector<TaskQueue*> &qvector, uint32_t batchSize, bool verbose = false) {
        cpp_workers.resize(_numCPPThreads);
        for(auto& w : cpp_workers)
            w = std::make_unique<WorkerCPU>(qvector[0], verbose, 0, batchSize);
    }
    
    void initCPPWorkersPerCPU(std::vector<TaskQueue*> &qvector, std::vector<int> numaDomains, uint32_t batchSize, bool verbose = false, int numQueues = 0, int queueMode = 0, int stealLogic = 0, bool pinWorkers = 0) {
        cpp_workers.resize(_numCPPThreads);
        if( numQueues == 0 ) {
            std::cout << "numQueues is 0, this should not happen." << std::endl;
        }
        //get_topology(topologyPhysicalIds, topologyUniqueThreads);
        
        int i = 0;
        for( auto& w : cpp_workers ) {
            w = std::make_unique<WorkerCPUPerCPU>(qvector, topologyPhysicalIds, topologyUniqueThreads, verbose, 0, batchSize, i, numQueues, queueMode, this->_stealLogic, pinWorkers);
            i++;
        }
    }
    
    void initCPPWorkersPerGroup(std::vector<TaskQueue*> &qvector, std::vector<int> numaDomains, uint32_t batchSize, bool verbose = false, int numQueues = 0, int queueMode = 0, int stealLogic = 0, bool pinWorkers = 0) {
        cpp_workers.resize(_numCPPThreads);
        if (numQueues == 0) {
            std::cout << "numQueues is 0, this should not happen." << std::endl;
        }
        //get_topology(topologyPhysicalIds, topologyUniqueThreads);
        if( _numCPPThreads < topologyUniqueThreads.size() )
            topologyUniqueThreads.resize(_numCPPThreads);
        int i = 0;
        for(auto& w : cpp_workers) {
            w = std::make_unique<WorkerCPUPerGroup>(qvector, topologyPhysicalIds, topologyUniqueThreads, verbose, 0, batchSize, i, numQueues, queueMode, this->_stealLogic, pinWorkers);
            i++;
        }
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
        _queueMode = 0;
        _numQueues = 1;
        _stealLogic = ctx->getUserConfig().victimSelection;
        get_topology(topologyPhysicalIds, topologyUniqueThreads);
        if( std::thread::hardware_concurrency() < topologyUniqueThreads.size() && _ctx->config.hyperthreadingEnabled )
            topologyUniqueThreads.resize(_numCPPThreads);
        _totalNumaDomains = std::set<double>( topologyPhysicalIds.begin(), topologyPhysicalIds.end() ).size();
        if( _ctx->config.debugMultiThreading ) {
            for(const auto & topologyEntry: topologyPhysicalIds) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl;
            for(const auto & topologyEntry: topologyUniqueThreads) {
                std::cout << topologyEntry << ',';
            }
            std::cout << std::endl;
            std::cout << "_totalNumaDomains=" << _totalNumaDomains << std::endl;
        }
        
        if (ctx->getUserConfig().queueSetupScheme == PERGROUP) {
            _queueMode = 1;
            _numQueues = _totalNumaDomains;
        } else if (ctx->getUserConfig().queueSetupScheme == PERCPU) {
            _queueMode = 2;
            _numQueues = _numThreads;
        }
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

    void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, DenseMatrix<VT>*** res, bool* isScalar, Structure** inputs,
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

    void executeCpuQueues(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res, bool* isScalar, Structure** inputs,
                            size_t numInputs, size_t numOutputs, const int64_t* outRows, const int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    void executeQueuePerCPU(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res, bool* isScalar, Structure** inputs,
                            size_t numInputs, size_t numOutputs, const int64_t* outRows, const int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    [[maybe_unused]] void executeQueuePerDeviceType(std::vector<std::function<PipelineFunc>> funcs, CSRMatrix<VT>*** res, bool* isScalar, Structure** inputs,
                            size_t numInputs, size_t numOutputs, int64_t* outRows, int64_t* outCols,
                            VectorSplit* splits, VectorCombine* combines, DCTX(ctx), bool verbose);

    void combineOutputs(CSRMatrix<VT>***& res, CSRMatrix<VT>***& res_cuda, [[maybe_unused]] size_t numOutputs,
                        [[maybe_unused]] mlir::daphne::VectorCombine* combines) override {}
};
