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

#ifndef SRC_RUNTIME_LOCAL_VECTORIZED_SCHEDULER_H
#define SRC_RUNTIME_LOCAL_VECTORIZED_SCHEDULER_H

#include <runtime/local/vectorized/TaskQueues.h>
#include <runtime/local/vectorized/Tasks.h>
#include <runtime/local/vectorized/Workers.h>
#include <ir/daphneir/Daphne.h>

#include <thread>
#include <functional>

template <class VT>
class Scheduler {

private:
    std::string mode;
    uint64_t rlen;
    uint32_t n_proc;
    DenseMatrix<VT>* res;
    DenseMatrix<VT>* input1;
    DenseMatrix<VT>* input2;
    uint64_t initChunkSize = 1;


public:
    Scheduler() : Scheduler(){};
    Scheduler(uint64_t rlen, uint32_t n_proc, DenseMatrix<VT>*& res, DenseMatrix<VT>* input1, DenseMatrix<VT>* input2, std::string mode){
        this.rlen = rlen;
        this.n_proc = n_proc;
        this.res = res;
        this.input1 = input1;
        this.input2 = input2;
        this.mode = mode;
    }

    void initChunk(uint32_t initChunk){
        this.initChunkSize = initChunk;
    }


    void run(void (*func)(DenseMatrix<VT>*,DenseMatrix<VT>*,DenseMatrix<VT>*)){
        TaskQueue* q = new BlockingTaskQueue(this.input1->getNumRows());
        uint64_t chunkSize = 1;
        uint64_t batchsize = 1;
        uint64_t remaining = 0;
        uint32_t k = 0;
        switch(this.mode){
            case "SCH_STATIC":
                //STATIC
                chunkSize = (uint64_t)ceil((double)rlen/n_proc/4);
                batchsize = 1; // row-at-a-time
                for(k=0; (k<n_proc*4) & (k*chunkSize<rlen); k++) {
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, this.res, this.input1, this.input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
                }
                q->closeInput();
                break;
            case "SCH_GSS":
                //GSS
                remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                chunkSize = 0;
               
                while (remaining >= n_proc){
                    //std::cout <<  chunkSize << " " <<remaining <<"\n";
                    chunkSize = (uint64_t)ceil((double)remaining/n_proc);
                    rlen = remaining;
                    remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, this.res, this.input1, this.input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
                
                    k++;
                }
                q->closeInput();
                break;

            case "SCH_TFSS":
                        //TFSS
                n_proc = 4;
                remaining = rlen - (uint64_t)ceil((double)rlen/n_proc);
                chunkSize = (uint64_t)ceil((double)rlen/n_proc);

                rlen = 100;
                chunkSize = 0;
                double steps  = ceil(2.0*rlen/(chunkSize+1)); //n=2N/f+l
                double tss_delta = (double) (chunkSize - 1)/(double) (steps-1);
            
                while(remaining >= n_proc){
                    //std::cout <<  chunkSize << " " <<remaining <<"\n";
                    chunkSize = ceil((double) rlen / ((double) 2*n_proc)); 
                    rlen = remaining;
                    remaining = rlen - tss_delta;
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, this.res, this.input1, this.input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
                    k++;
                }
                q->closeInput();
                break;
            case "SCH_FAC2":

                n_proc = 4;
                //dbl_parm1 = (double)rlen / (double)n_proc;

                //pr->u.p.parm1 = chunk;
                //pr->u.p.dbl_parm1 = dbl_parm1;
                break;
        
        }
    }

};

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_SCHEDULER_H