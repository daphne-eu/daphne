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
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs,
        const int64_t *outRows, const int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx),
        bool verbose) {
//     TODO: reduce code duplication
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    // ToDo: sparse output mem requirements
    auto row_mem = mem_required / len;

    // create task queue (w/o size-based blocking)
    std::unique_ptr<TaskQueue> q = std::make_unique<BlockingTaskQueue>(len);

    auto batchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8388608 / row_mem)));
    this->initCPPWorkers(q.get(), batchSize8M, verbose);


    for(size_t i = 0; i < numOutputs; i++)
        if(*(res[i]) != nullptr)
            throw std::runtime_error("TODO");

    std::vector<VectorizedDataSink<CSRMatrix<VT>> *> dataSinks(numOutputs);
    for(size_t i = 0; i < numOutputs; i++)
        dataSinks[i] = new VectorizedDataSink<CSRMatrix<VT>>(combines[i], outRows[i], outCols[i]);

    // lock for aggregation combine
    // TODO: multiple locks per output

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    int method=ctx->config.taskPartitioningScheme;
    int chunkParam = ctx->config.minimumTaskSize;
    if(chunkParam<=0)
        chunkParam=1;
    LoadPartitioning lp(method, len, chunkParam, this->_numThreads, false);
    while (lp.hasNextChunk()) {
        endChunk += lp.getNextChunk();
        q->enqueueTask(new CompiledPipelineTask<CSRMatrix<VT>>(CompiledPipelineTaskData<CSRMatrix<VT>>{funcs, isScalar,
                inputs, numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows,
                outCols, 0, ctx}, dataSinks));
        startChunk = endChunk;
    }
    q->closeInput();

    this->joinAll();
    for(size_t i = 0; i < numOutputs; i++) {
        *(res[i]) = dataSinks[i]->consume();
        delete dataSinks[i];
    }
}

template<typename VT>
[[maybe_unused]] void MTWrapper<CSRMatrix<VT>>::executeQueuePerDeviceType(std::vector<std::function<void(CSRMatrix<VT> ***, Structure **,
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, bool* isScalar, Structure **inputs, size_t numInputs, size_t numOutputs,
        int64_t *outRows, int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    throw std::runtime_error("sparse multi queue vect exec not implemented");
}

template class MTWrapper<CSRMatrix<double>>;
template class MTWrapper<CSRMatrix<float>>;