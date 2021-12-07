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

#ifndef SRC_RUNTIME_LOCAL_VECTORIZED_MTWRAPPER_H
#define SRC_RUNTIME_LOCAL_VECTORIZED_MTWRAPPER_H

#include <runtime/local/vectorized/TaskQueues.h>
#include <runtime/local/vectorized/Tasks.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>
#include <runtime/local/vectorized/Workers.h>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <ir/daphneir/Daphne.h>

#include <thread>
#include <functional>
#include <queue>

//TODO use the wrapper to cache threads
//TODO generalize for arbitrary inputs (not just binary)

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;
template <class DT>
class MTWrapper {};

template <typename VT>
class MTWrapper<DenseMatrix<VT>> {
private:
    uint32_t _numThreads;

public:
    MTWrapper() : MTWrapper(std::thread::hardware_concurrency()) {}
    MTWrapper(uint32_t numThreads) {
        _numThreads = (numThreads <= 0) ? 32 : numThreads;
    }
    ~MTWrapper() = default;

    void execute(std::function<void(DenseMatrix<VT> ***, Structure **)> func,
                 DenseMatrix<VT> *&res,
                 Structure **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 int64_t *outRows,
                 int64_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines,
                 bool verbose)
    {
        uint64_t len = 0;
        // due to possible broadcasting we have to check all inputs
        for (auto i = 0u; i < numInputs; ++i) {
            if (splits[i] == mlir::daphne::VectorSplit::ROWS)
                len = std::max(len, inputs[i]->getNumRows());
        }
        // create task queue (w/o size-based blocking)
        TaskQueue* q = new BlockingTaskQueue(len); // again here is the maximum possible number of tasks
        if(const char* env_m = std::getenv("DAPHNE_THREADS")){
            _numThreads= std::stoi(env_m);
        }
        // create workers threads
        WorkerCPU* workers[_numThreads];
        std::thread workerThreads[_numThreads];
        for(uint32_t i=0; i<_numThreads; i++) {
            workers[i] = new WorkerCPU(q, verbose);
            workerThreads[i] = std::thread(runWorker, workers[i]);
        }

        assert(numOutputs == 1 && "TODO");
        // output allocation for row-wise combine
        if(res == nullptr && outRows[0] != -1 && outCols[0] != -1) {
            auto zeroOut = combines[0] == mlir::daphne::VectorCombine::ADD;
            res = DataObjectFactory::create<DenseMatrix<VT>>(outRows[0], outCols[0], zeroOut);
        }
        // lock for aggregation combine
        std::mutex resLock;

        // create tasks and close input
        uint64_t startChunk = 0;
        uint64_t endChunk = 0;
        uint64_t batchsize = 100; // 100-rows-at-a-time
        uint64_t chunkParam = 1;
        LoadPartitioning lp(STATIC, len, chunkParam,_numThreads,false);
        while(lp.hasNextChunk()){
            endChunk += lp.getNextChunk();
            q->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(CompiledPipelineTaskData<DenseMatrix<VT>>{
                    func,
                    inputs,
                    numInputs,
                    numOutputs,
                    outRows,
                    outCols,
                    splits,
                    combines,
                    startChunk,
                    endChunk,
                    batchsize,
                    outRows[0],
                    outCols[0]},
                resLock,
                res
            ));
            startChunk = endChunk;
        }
        q->closeInput();

        // barrier (wait for completed computation)
        for(uint32_t i=0; i<_numThreads; i++)
            workerThreads[i].join();

        // cleanups
        for(uint32_t i=0; i<_numThreads; i++)
            delete workers[i];
        delete q;
    }
};

template<typename VT>
class MTWrapper<CSRMatrix<VT>> {
private:
    uint32_t _numThreads;

public:
    MTWrapper() : MTWrapper(std::thread::hardware_concurrency()) {}
    MTWrapper(uint32_t numThreads) {
        _numThreads = (numThreads <= 0) ? 32 : numThreads;
    }
    ~MTWrapper() = default;

    void execute(std::function<void(CSRMatrix<VT> ***, Structure **)> func,
                 CSRMatrix<VT> *&res,
                 Structure **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 int64_t *outRows,
                 int64_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines,
                 bool verbose) {
        // TODO: reduce code duplication
        uint64_t len = 0;
        // due to possible broadcasting we have to check all inputs
        for(auto i = 0u ; i < numInputs ; ++i) {
            if(splits[i] == mlir::daphne::VectorSplit::ROWS)
                len = std::max(len, inputs[i]->getNumRows());
        }
        // create task queue (w/o size-based blocking)
        TaskQueue *q = new BlockingTaskQueue(len); // again here is the maximum possible number of tasks
        if(const char *env_m = std::getenv("DAPHNE_THREADS")) {
            _numThreads = std::stoi(env_m);
        }
        // create workers threads
        WorkerCPU *workers[_numThreads];
        std::thread workerThreads[_numThreads];
        for(uint32_t i = 0 ; i < _numThreads ; i++) {
            workers[i] = new WorkerCPU(q, verbose);
            workerThreads[i] = std::thread(runWorker, workers[i]);
        }

        assert(numOutputs == 1 && "TODO");
        assert(res == nullptr && "TODO");
        VectorizedDataSink<CSRMatrix<VT>> dataSink(combines[0], outRows[0], outCols[0]);

        // create tasks and close input
        uint64_t startChunk = 0;
        uint64_t endChunk = 0;
        uint64_t batchsize = 100; // 100-rows-at-a-time
        uint64_t chunkParam = 1;
        LoadPartitioning lp(STATIC, len, chunkParam, _numThreads, false);
        while(lp.hasNextChunk()) {
            endChunk += lp.getNextChunk();
            q->enqueueTask(new CompiledPipelineTask<CSRMatrix<VT>>(CompiledPipelineTaskData<CSRMatrix<VT>>{
                    func,
                    inputs,
                    numInputs,
                    numOutputs,
                    outRows,
                    outCols,
                    splits,
                    combines,
                    startChunk,
                    endChunk,
                    batchsize,
                    outRows[0],
                    outCols[0]},
                dataSink
            ));
            startChunk = endChunk;
        }
        q->closeInput();

        // barrier (wait for completed computation)
        for(uint32_t i = 0 ; i < _numThreads ; i++)
            workerThreads[i].join();
        res = dataSink.consume();

        // cleanups
        for(uint32_t i = 0 ; i < _numThreads ; i++)
            delete workers[i];
        delete q;
    }
};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_MTWRAPPER_H
