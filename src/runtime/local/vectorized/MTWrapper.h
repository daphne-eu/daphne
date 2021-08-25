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

#include <thread>
#include <runtime/local/vectorized/TaskQueues.h>
#include <runtime/local/vectorized/Tasks.h>
#include <runtime/local/vectorized/Workers.h>
#include <functional>

//TODO use the wrapper to cache threads
//TODO generalize for arbitrary inputs (not just binary)

template <class VT>
class MTWrapper {
private:
    uint32_t _numThreads;

public:
    MTWrapper() : MTWrapper(std::thread::hardware_concurrency()) {}
    MTWrapper(uint32_t numThreads) {
        _numThreads = (numThreads <= 0) ? 32 : numThreads;
    }
    ~MTWrapper() = default;

    void execute(void (*func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*),
        DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2)
    {
        execute(func, res, input1, input2, false);
    }

    void execute(void (*func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*),
        DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2, bool verbose)
    {
        // create task queue (w/o size-based blocking)
        TaskQueue* q = new BlockingTaskQueue(input1->getNumRows());

        // create workers threads
        WorkerCPU* workers[_numThreads];
        std::thread workerThreads[_numThreads];
        for(uint32_t i=0; i<_numThreads; i++) {
            workers[i] = new WorkerCPU(q, verbose);
            workerThreads[i] = std::thread(runWorker, workers[i]);
        }

        // output allocation (currently only according to input shape only)
        if( res == nullptr )
            res = DataObjectFactory::create<DenseMatrix<VT>>(input1->getNumRows(), input1->getNumCols(), false);

        // create tasks and close input
        // TODO UNIBAS - integration hook scheduling
        uint64_t rlen = input1->getNumRows();
        uint64_t blksize = (uint64_t)ceil((double)rlen/_numThreads/4);
        uint64_t batchsize = 1; // row-at-a-time
        for(uint32_t k=0; (k<_numThreads*4) & (k*blksize<rlen); k++) {
            q->enqueueTask(new SingleOpTask<VT>(
                func, res, input1, input2, k*blksize, std::min((k+1)*blksize,rlen), batchsize));
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

    void execute(std::function<void(DenseMatrix<VT>***, DenseMatrix<VT>**)> func,
                 DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2)
    {
        execute(func, res, input1, input2, false);
    }

    void execute(std::function<void(DenseMatrix<VT>***, DenseMatrix<VT>**)> func,
                 DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2, bool verbose)
    {
        // create task queue (w/o size-based blocking)
        TaskQueue* q = new BlockingTaskQueue(input1->getNumRows());

        // create workers threads
        WorkerCPU* workers[_numThreads];
        std::thread workerThreads[_numThreads];
        for(uint32_t i=0; i<_numThreads; i++) {
            workers[i] = new WorkerCPU(q, verbose);
            workerThreads[i] = std::thread(runWorker, workers[i]);
        }

        // output allocation (currently only according to input shape only)
        if( res == nullptr )
            res = DataObjectFactory::create<DenseMatrix<VT>>(input1->getNumRows(), input1->getNumCols(), false);

        // create tasks and close input
        // TODO UNIBAS - integration hook scheduling
        uint64_t rlen = input1->getNumRows();
        // every thread gets 4 tasks (computation is rounded up)
        uint64_t blksize = (((rlen - 1) / _numThreads + 1) - 1) / 4 + 1;
        uint64_t batchsize = 1; // row-at-a-time
        for(uint32_t k = 0; k * blksize < rlen; k++) {
            q->enqueueTask(new CompiledPipelineTask<VT>(
                func, res, input1, input2, k * blksize, std::min((k + 1) * blksize, rlen), batchsize));
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

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_MTWRAPPER_H
