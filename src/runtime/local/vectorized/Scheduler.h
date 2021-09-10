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

#ifndef SRC_RUNTIME_LOCAL_SCHEDULER_H
#define SRC_RUNTIME_LOCAL_SCHEDULER_H

#include <runtime/local/vectorized/TaskQueues.h>
#include <runtime/local/vectorized/Tasks.h>
#include <runtime/local/vectorized/Workers.h>
#include <ir/daphneir/Daphne.h>

#include <thread>
#include <functional>

class Scheduler {

private:
    auto mode;
    uint64_t rlen;
    uint32_t n_proc;
    DenseMatrix<VT>* input1;
    DenseMatrix<VT>* input2;
    initChunkSize = 1;

public:
    Scheduler() : Scheduler(){};
    Scheduler(uint64_t rlen, uint32_t n_proc, DenseMatrix<VT>* input1, DenseMatrix<VT>*, auto mode){
        this.rlen = rlen;
        this.n_proc = n_proc;
        this.input1 = input1;
        this.input2 = input2;
        this.mode = mode;
    }

    void initChunk(uint32_t initChunk){
        this.initChunkSize = initChunk;
    }

    void run(){
        switch(this.mode){
            case "SCH_STATIC":
                //STATIC
                uint64_t chunkSize = (uint64_t)ceil((double)rlen/_numThreads/4);
                uint64_t batchsize = 1; // row-at-a-time
                for(uint32_t k=0; (k<_numThreads*4) & (k*chunkSize<rlen); k++) {
                    q->enqueueTask(new SingleOpTask<VT>(
                        func, res, input1, input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
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
                        func, res, input1, input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
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
                        func, res, input1, input2, k*chunkSize, std::min((k+1)*chunkSize,rlen), batchsize));
                    
                    q->closeInput();
                }
                break;
            case "SCH_FAC2":

                uint64_t n_proc = 4;
                dbl_parm1 = (double)rlen / (double)n_proc;

                pr->u.p.parm1 = chunk;
                pr->u.p.dbl_parm1 = dbl_parm1;
                break;
        
        }
    }




    

};