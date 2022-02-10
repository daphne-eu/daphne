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

template<typename VT>
void MTWrapper<CSRMatrix<VT>>::executeSingleQueue(std::vector<std::function<void(CSRMatrix<VT> ***, Structure **,
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, Structure **inputs, size_t numInputs, size_t numOutputs,
        int64_t *outRows, int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    // TODO: reduce code duplication
    uint64_t len = 0;
    // due to possible broadcasting we have to check all inputs
    for(auto i = 0u ; i < numInputs ; ++i) {
        if(splits[i] == mlir::daphne::VectorSplit::ROWS)
            len = std::max(len, inputs[i]->getNumRows());
    }
    // create task queue (w/o size-based blocking)
    TaskQueue *q = new BlockingTaskQueue(len); // again here is the maximum possible number of tasks
    // create workers threads
    WorkerCPU *workers[this->_numThreads];
    std::thread workerThreads[this->_numThreads];
    for(uint32_t i = 0 ; i < this->_numThreads ; i++) {
        workers[i] = new WorkerCPU(q, verbose);
        workerThreads[i] = std::thread(runWorker, workers[i]);
    }

    assert(numOutputs == 1 && "TODO");
    assert(*(res[0]) == nullptr && "TODO");
    VectorizedDataSink<CSRMatrix<VT>> dataSink(combines[0], outRows[0], outCols[0]);

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    uint64_t chunkParam = 1;
    LoadPartitioning lp(STATIC, len, chunkParam, this->_numThreads, false);
    while(lp.hasNextChunk()) {
        endChunk += lp.getNextChunk();
        q->enqueueTask(new CompiledPipelineTask<CSRMatrix<VT>>(CompiledPipelineTaskData<CSRMatrix<VT>> {funcs, inputs,
                numInputs, numOutputs, outRows, outCols, splits, combines, startChunk, endChunk, outRows, outCols},
                dataSink));
        startChunk = endChunk;
    }
    q->closeInput();

    // barrier (wait for completed computation)
    for(uint32_t i = 0 ; i < this->_numThreads ; i++)
        workerThreads[i].join();
    *(res[0]) = dataSink.consume();

    // cleanups
    for(uint32_t i = 0 ; i < this->_numThreads ; i++)
        delete workers[i];
    delete q;
}

template<typename VT>
void MTWrapper<CSRMatrix<VT>>::executeQueuePerDeviceType(std::vector<std::function<void(CSRMatrix<VT> ***, Structure **,
        DCTX(ctx))>> funcs, CSRMatrix<VT> ***res, Structure **inputs, size_t numInputs, size_t numOutputs,
        int64_t *outRows, int64_t *outCols, VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    assert("sparse multi queue vect exec not implemented");
}

template class MTWrapper<CSRMatrix<double>>;
template class MTWrapper<CSRMatrix<float>>;