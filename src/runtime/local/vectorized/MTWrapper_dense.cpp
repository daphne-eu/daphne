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

template<typename VT>
[[maybe_unused]] void MTWrapper<DenseMatrix<VT>>::executeCpuQueues(
        std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
        const bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
        VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    auto inputProps = getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;

    std::vector<std::unique_ptr<TaskQueue>> q;
    std::vector<TaskQueue*> qvector;
    if (ctx->getUserConfig().pinWorkers) {
        for(int i=0; i<this->_numQueues; i++) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
            std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
            q.push_back(std::move(tmp));
            qvector.push_back(q[i].get());
        }
    } else {
        for(int i=0; i<this->_numQueues; i++) {
            std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
            q.push_back(std::move(tmp));
            qvector.push_back(q[i].get());
        }
    }

    auto batchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8388608 / row_mem)));
    this->initCPPWorkers(qvector, batchSize8M, verbose, this->_numQueues, this->_queueMode, ctx->getUserConfig().pinWorkers);

    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    uint64_t currentItr = 0;
    uint64_t target;
    int method=ctx->config.taskPartitioningScheme;
    int chunkParam = ctx->config.minimumTaskSize;
    if(chunkParam<=0)
        chunkParam=1;
    if (ctx->getUserConfig().prePartitionRows) {
        uint64_t oneChunk = len/this->_numQueues;
        int remainder = len - (oneChunk * this->_numQueues);
        std::vector<LoadPartitioning> lps;
        lps.emplace_back(method, oneChunk+remainder, chunkParam, this->_numThreads, false);
        for(int i=1; i<this->_numQueues; i++) {
            lps.emplace_back(method, oneChunk, chunkParam, this->_numThreads, false);
        }
        if (ctx->getUserConfig().pinWorkers) {
            for(int i=0; i<this->_numQueues; i++) {
                while (lps[i].hasNextChunk()) {
                    endChunk += lps[i].getNextChunk();
                    qvector[i]->enqueueTaskPinned(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar,
                            inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                            outCols, 0, ctx}, resLock, res), this->topologyResponsibleThreads[i]);
                    startChunk = endChunk;
                }
            }
        } else {
            for(int i=0; i<this->_numQueues; i++) {
                while (lps[i].hasNextChunk()) {
                    endChunk += lps[i].getNextChunk();
                    qvector[i]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar,
                            inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                            outCols, 0, ctx}, resLock, res));
                    startChunk = endChunk;
                }
            }
        }
    } else {
        LoadPartitioning lp(method, len, chunkParam, this->_numThreads, false);
        if (ctx->getUserConfig().pinWorkers) {
            while (lp.hasNextChunk()) {
                endChunk += lp.getNextChunk();
                target = currentItr % this->_numQueues;
                qvector[target]->enqueueTaskPinned(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar,
                        inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                        outCols, 0, ctx}, resLock, res), this->topologyUniqueThreads[target]);
                startChunk = endChunk;
		currentItr++;
            }
        } else {
            while (lp.hasNextChunk()) {
                endChunk += lp.getNextChunk();
                target = currentItr % this->_numQueues;
                qvector[target]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar,
                        inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                        outCols, 0, ctx}, resLock, res));
                startChunk = endChunk;
		currentItr++;
            }
        }
    }
    for(int i=0; i<this->_numQueues; i++) {
        qvector[i]->closeInput();
    }

    this->joinAll();
}

template class MTWrapper<DenseMatrix<double>>;
template class MTWrapper<DenseMatrix<float>>;
template class MTWrapper<DenseMatrix<int64_t>>;
