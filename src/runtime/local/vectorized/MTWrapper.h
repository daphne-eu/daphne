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
#include <runtime/local/vectorized/Workers.h>
#include <ir/daphneir/Daphne.h>

#include <thread>
#include <functional>

//TODO use the wrapper to cache threads
//TODO generalize for arbitrary inputs (not just binary)

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

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
        DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2, bool verbose, auto mode)
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
        switch(mode){
            case "SCH_STATIC":
                //STATIC
                uint64_t blksize = (uint64_t)ceil((double)rlen/_numThreads/4);
                uint64_t batchsize = 1; // row-at-a-time
                for(uint32_t k=0; (k<_numThreads*4) & (k*blksize<rlen); k++) {
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, res, input1, input2, k*blksize, std::min((k+1)*blksize,rlen), batchsize));
                }
                q->closeInput();
                break;
            case "SCH_GSS":
                //GSS
                uint64_t n_proc = 4;
                uint64_t remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                uint64_t chunkSize = 0;
                while (remaining >= n_proc){
                    //std::cout <<  chunkSize << " " <<remaining <<"\n";
                    chunkSize = (uint64_t)ceil((double)remaining/n_proc);
                    rlen = remaining;
                    remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, res, input1, input2, k*chunkSize, std::min
                    ));

                    q->closeInput();
                    
                }
                break;

            case "SCH_TFSS":
                        //TFSS
                uint64_t n_proc = 4;
                uint64_t remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                uint64_t chunkSize = (uint64_t)ceil((double)rlen/n_proc);

                uint64_t rlen = 100;
                uint64_t chunkSize = 0;
                double steps  = ceil(2.0*rlen/(chunkSize+1)); //n=2N/f+l
                double tss_delta = (double) (chunkSize - 1)/(double) (steps-1);
            
                while(remaining >= num_proc){
                    //std::cout <<  chunkSize << " " <<remaining <<"\n";
                    chunkSize = ceil((double) rlen / ((double) 2*n_proc)); 
                    rlen = remaining;
                    remaining = rlen - tss_delta;
                    q->enqueueTask(new SingleOpTask<VT>(new SingleOpTask<VT>(
                        func, res, input1, input2, k*chunkSize, std::min));
                    
                    q->closeInput();
                }
                break;

            case "SCH_FAC":
                double sigma = means_sigmas.at(loc->psource).at(2*(current_index.at(loc->psource)/(int)nproc)+1);  //this sigma is based on profiling
                T mu = means_sigmas.at(loc->psource).at(2*(current_index.at(loc->psource)/(int)nproc));
                current_index.at(loc->psource)++;

                dbl_parm1 = ((double)P * sigma) / (2.0 * mu);
                double b_0 = dbl_parm1 * 1 / sqrt(N); // initial b
                double x_0 = 1 + pow(b_0, 2.0) + b_0 * sqrt(pow(b_0, 2.0) + 2);
                parm1 = ceil(N / (x_0 * P));
                break;
        }


        // barrier (wait for completed computation)
        for(uint32_t i=0; i<_numThreads; i++)
            workerThreads[i].join();

        // cleanups
        for(uint32_t i=0; i<_numThreads; i++)
            delete workers[i];
        delete q;
    }

    void execute(std::function<void(DenseMatrix<VT>***, DenseMatrix<VT>**)> func,
                 DenseMatrix<VT> *&res, DenseMatrix<VT> **inputs, size_t numInputs)
    {
        execute(func, res, inputs, numInputs, false);
    }

    void execute(std::function<void(DenseMatrix<VT> ***, DenseMatrix<VT> **)> func,
                 DenseMatrix<VT> *&res,
                 DenseMatrix<VT> **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 int64_t *outRows,
                 int64_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines,
                 bool verbose)
    {
        auto numTasks = _numThreads * 4;
        // create task queue (w/o size-based blocking)
        TaskQueue* q = new BlockingTaskQueue(numTasks);

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
        // TODO UNIBAS - integration hook scheduling
        uint64_t len = 0;
        // due to possible broadcasting we have to check all inputs
        for (auto i = 0u; i < numInputs; ++i) {
            if (splits[i] == mlir::daphne::VectorSplit::ROWS)
                len = std::max(len, inputs[i]->getNumRows());
        }
        uint64_t blksize = (len - 1) / numTasks + 1; // integer ceil
        uint64_t batchsize = 100; // 100-rows-at-a-time
        for(uint32_t k = 0; k * blksize < len; k++) {
            q->enqueueTask(new CompiledPipelineTask<VT>(
                func,
                resLock,
                res,
                inputs,
                numInputs,
                numOutputs,
                outRows,
                outCols,
                splits,
                combines,
                k * blksize,
                std::min((k + 1) * blksize, len),
                batchsize));
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
