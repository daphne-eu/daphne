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
void MTWrapper<CSRMatrix<VT>>::executeSingleQueue(std::vector<std::function<void(CSRMatrix<VT> ***, Structure **,
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, Structure **inputs, size_t numInputs, size_t numOutputs,
        const int64_t *outRows, const int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx),
        bool verbose) {
//     TODO: reduce code duplication
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    // ToDo: sparse output mem requirements
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

    // create task queue (w/o size-based blocking)
    std::vector<std::unique_ptr<TaskQueue>> q;
    std::vector<TaskQueue*> qvector;
    for(uint32_t i=0; i<this->_numThreads; i++) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
        q.emplace_back(new BlockingTaskQueue(len));
        q.push_back(std::move(tmp));
        qvector.push_back(q[i].get());
    }

    auto batchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8388608 / row_mem)));
    this->initCPPWorkers(qvector, batchSize8M, verbose, _numDeques, queueMode);

    assert(numOutputs == 1 && "TODO");
    assert(*(res[0]) == nullptr && "TODO");
    VectorizedDataSink<CSRMatrix<VT>> dataSink(combines[0], outRows[0], outCols[0]);

    // lock for aggregation combine
    // TODO: multiple locks per output

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
        q[target]->enqueueTask(new CompiledPipelineTask<CSRMatrix<VT>>(CompiledPipelineTaskData<CSRMatrix<VT>>{funcs,
                inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                outCols, 0, ctx}, dataSink));
        startChunk = endChunk;
        currentItr++;
    }
    for(int i=0; i<_numDeques; i++) {
        q[i]->closeInput();
    }

    this->joinAll();
    *(res[0]) = dataSink.consume();
}

template<typename VT>
[[maybe_unused]] void MTWrapper<CSRMatrix<VT>>::executeQueuePerDeviceType(std::vector<std::function<void(CSRMatrix<VT> ***, Structure **,
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, Structure **inputs, size_t numInputs, size_t numOutputs,
        int64_t *outRows, int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    throw std::runtime_error("sparse multi queue vect exec not implemented");
}

template class MTWrapper<CSRMatrix<double>>;
template class MTWrapper<CSRMatrix<float>>;
