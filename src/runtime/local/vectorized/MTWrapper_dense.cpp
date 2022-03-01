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

#include "MTWrapper.h"
#include <runtime/local/vectorized/Tasks.h>
#ifdef USE_CUDA
#include <runtime/local/vectorized/TasksCUDA.h>
#endif

template<typename VT>
void MTWrapper<DenseMatrix<VT>>::executeSingleQueue(
        std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
        Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
        VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += this->allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;

    int queueMode = 0;
    int _numDeques;
    if(queueMode == 0) {
        // One centralized queue
        _numDeques = 1;
    } else if (queueMode == 1) {
        // One queue per socket (or group)
        std::cout << "Not supported yet." << std::endl;
    } else if (queueMode == 2) {
        // One queue per thread
        _numDeques = this->_numThreads;
    }
//std::cout << "_numDeques: " << _numDeques << std::endl;


    // create task queue (w/o size-based blocking)
    std::vector<std::unique_ptr<TaskQueue>> q;
    std::vector<TaskQueue*> qvector;
    for(uint32_t i=0; i<this->_numThreads; i++) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
        //q.emplace_back(new BlockingTaskQueue(len));
        q.push_back(std::move(tmp));
        qvector.push_back(q[i].get());
    }
    //std::unique_ptr<TaskQueue> q = std::make_unique<BlockingTaskQueue>(len);

    auto batchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8388608 / row_mem)));
    this->initCPPWorkers(qvector, batchSize8M, verbose, _numDeques, queueMode);
#ifdef USE_CUDA
    if(this->_numCUDAThreads) {
        this->initCUDAWorkers(q.get(), batchSize8M * 4, verbose);
        this->cudaPrefetchInputs(inputs, numInputs, mem_required, splits);
#ifndef NDEBUG
        std::cout << "Required memory (ins/outs): " << mem_required << "\nRequired mem/row: " << row_mem << std::endl;
        std::cout << "batchsizeCPU=" << batchSize8M << " batchsizeGPU=" << batchSize8M*4 << std::endl;
#endif
    }
#endif

    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    uint64_t currentItr = 0;
    uint64_t target;
    auto chunkParam = 1;
    LoadPartitioning lp(STATIC, len, chunkParam, this->_numThreads, false);
    while (lp.hasNextChunk()) {
        endChunk += lp.getNextChunk();
        target = currentItr%_numDeques;
        q[target]->enqueueTaskOnTarget(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{funcs,
                inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                outCols, 0, ctx}, resLock, res), target);
        startChunk = endChunk;
        currentItr++;
    }
    //std::cout << "len is " << len << std::endl;
    //std::cout << "Total Itr: " << currentItr << std::endl;
    //q[0]->closeInput();
    for(int i=0; i<_numDeques; i++) {
        q[i]->closeInput();
    }

    this->joinAll();
}

template<typename VT>
[[maybe_unused]] void MTWrapper<DenseMatrix<VT>>::executeQueuePerDeviceType(
        std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
        Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
        VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    size_t device_task_len = 0ul;
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += this->allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;
    auto batchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8388608 / row_mem)));
    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

    int queueMode = 0;
    int _numDeques;
    if(queueMode == 0) {
        _numDeques = 1;
    } else if (queueMode == 1) {
        //One queue per socket (or group)
        std::cout << "Not supported yet." << std::endl;
    } else if (queueMode == 2) {
        _numDeques = this->_numThreads;
    }

#ifdef USE_CUDA
    // ToDo: multi-device support :-P
    float taskRatioCUDA = 0.25f;
    auto gpu_task_len = static_cast<size_t>(std::ceil(static_cast<float>(len) * taskRatioCUDA));
    device_task_len += gpu_task_len;
    std::unique_ptr<TaskQueue> q_cuda = std::make_unique<BlockingTaskQueue>(gpu_task_len);
    this->initCUDAWorkers(q_cuda.get(), batchSize8M * 4, verbose);

    auto*** res_cuda = new DenseMatrix<VT>**[numOutputs];
    //    auto blksize = static_cast<size_t>(std::floor(cctx->getMemBudget() / row_mem));
    auto blksize = gpu_task_len / ctx->cuda_contexts.size();
#ifndef NDEBUG
    std::cout << "gpu_task_len:  " << gpu_task_len << "\ntaskRatioCUDA: " << taskRatioCUDA << "\nBlock size: "
              << blksize << std::endl;
#endif
    for (size_t i = 0; i < numOutputs; ++i) {
        res_cuda[i] = new DenseMatrix<VT>*;
        if(combines[i] == mlir::daphne::VectorCombine::ROWS) {
            auto rc2 = static_cast<DenseMatrix<VT> *>((*res[i]))->slice(0, gpu_task_len);
            (*res_cuda[i]) = rc2;
        }
        else if(combines[i] == mlir::daphne::VectorCombine::COLS) {
            (*res_cuda[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->slice(0, outRows[i], 0, gpu_task_len);
        }
        else {
            (*res_cuda[i]) = (*res[i]);
        }
    }

    for (uint32_t k = 0; k < gpu_task_len; k += blksize) {
        q_cuda->enqueueTask(new CompiledPipelineTaskCUDA<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{
                funcs, inputs, numInputs, numOutputs, outRows, outCols, splits, combines, k, std::min(k + blksize, len),
                outRows, outCols, 0, ctx}, resLock, res_cuda));
    }
    q_cuda->closeInput();
#endif

    auto cpu_task_len = len - device_task_len;
    DenseMatrix<VT> ***res_cpp{};
    std::vector<std::unique_ptr<TaskQueue>> q_cpp;
    std::vector<TaskQueue*> q_cppvector;
    if(cpu_task_len > 0) {
        for(uint32_t i=0; i<this->_numThreads; i++) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
            std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
            q_cpp.push_back(std::move(tmp));
            q_cppvector.push_back(q_cpp[i].get());
            this->initCPPWorkers(q_cppvector, batchSize8M, verbose, _numDeques, queueMode);
        }
            res_cpp = new DenseMatrix<VT> **[numOutputs];
            auto offset = device_task_len;
        for (size_t i = 0; i < numOutputs; ++i) {
            res_cpp[i] = new DenseMatrix<VT> *;
            if(combines[i] == mlir::daphne::VectorCombine::ROWS) {
                (*res_cpp[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->slice(device_task_len, len);
            }
            else if(combines[i] == mlir::daphne::VectorCombine::COLS) {
                (*res_cpp[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->slice(0, outRows[i], device_task_len, len);
            }
            else {
                (*res_cpp[i]) = (*res[i]);
            }
        }

        uint64_t startChunk = device_task_len;
        uint64_t endChunk = device_task_len;
        uint64_t currentItr = 0;
        uint64_t target;
        auto chunkParam = 1;
        LoadPartitioning lp(STATIC, cpu_task_len, chunkParam, this->_numCPPThreads, false);
        while (lp.hasNextChunk()) {
            endChunk += lp.getNextChunk();
            target = currentItr%_numDeques;
            q_cpp[target]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{
                    funcs, inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk,
                    outRows, outCols, offset, ctx}, resLock, res_cpp));
            startChunk = endChunk;
            currentItr++;
        }
        for(int i=0; i<_numDeques; i++) {
            q_cpp[i]->closeInput();
        }
    }
    this->joinAll();

#ifdef USE_CUDA
    this->combineOutputs(res, res_cuda, numOutputs, combines);
#endif

    if(cpu_task_len > 0) {
        for (size_t i = 0; i < numOutputs; ++i) {
            if(combines[i] == mlir::daphne::VectorCombine::ROWS || combines[i] == mlir::daphne::VectorCombine::COLS)
                DataObjectFactory::destroy((*res_cpp[i]));
        }
    }
}

#ifdef USE_CUDA
template<typename VT>
void MTWrapper<DenseMatrix<VT>>::combineOutputs(DenseMatrix<VT>***& res_, DenseMatrix<VT>***& res_cuda_, size_t numOutputs,
                                                mlir::daphne::VectorCombine* combines) {
    for (size_t i = 0; i < numOutputs; ++i) {
        auto* res = static_cast<DenseMatrix<VT> *>((*res_[i]));
        auto* res_cuda = static_cast<DenseMatrix<VT> *>((*res_cuda_[i]));
        if (combines[i] == mlir::daphne::VectorCombine::ROWS) {
            const auto &const_res_cuda = *res_cuda;
            auto data_dest = res->getValues();
            CHECK_CUDART(cudaMemcpy(data_dest, const_res_cuda.getValuesCUDA(), const_res_cuda.bufferSize(),
                                    cudaMemcpyDeviceToHost));
            DataObjectFactory::destroy(res_cuda);
        }
        else if (combines[i] == mlir::daphne::VectorCombine::COLS) {
            const auto &const_res_cuda = *res_cuda;
            auto dst_base_ptr = res->getValues();
            auto src_base_ptr = const_res_cuda.getValuesCUDA();
            for(auto j = 0u; j < res_cuda->getNumRows(); ++j) {
                auto data_src = src_base_ptr + res_cuda->getRowSkip() * j;
                auto data_dst = dst_base_ptr + res->getRowSkip() * j;
                CHECK_CUDART(cudaMemcpy(data_dst, data_src,res_cuda->getNumCols() * sizeof(VT),
                                        cudaMemcpyDeviceToHost));
            }
            DataObjectFactory::destroy(res_cuda);
        }
    }
}
#else
template<typename VT>
void MTWrapper<DenseMatrix<VT>>::combineOutputs(DenseMatrix<VT>***& res_, DenseMatrix<VT>***& res_cuda_, size_t numOutputs,
        mlir::daphne::VectorCombine* combines) { }
#endif

template class MTWrapper<DenseMatrix<double>>;
template class MTWrapper<DenseMatrix<float>>;
