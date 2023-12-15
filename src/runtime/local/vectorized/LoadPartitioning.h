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

#pragma once

#include "LoadPartitioningDefs.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <iostream>

class LoadPartitioning {

private:
    int schedulingMethod;
    uint64_t totalTasks;
    uint64_t chunkParam;
    uint64_t scheduledTasks;
    uint64_t remainingTasks;
    uint32_t totalWorkers;
    uint64_t schedulingStep;
    uint64_t tssChunk; 
    uint64_t tssDelta;
    uint64_t mfscChunk;
    uint32_t fissStages;
    int getMethod (const char * method){
        return std::stoi(method);
    }
    int getStages(int tasks, int workers){
        int actual_step=0;
        int scheduled=0;
        int step=0;
        while (scheduled < tasks){
            actual_step=step/workers;
            double chunk = pow(0.5,actual_step+1)*tasks/float(workers);
            scheduled+=ceil(chunk);
            step+=1;
        }
        return actual_step+1;
    }
public:
LoadPartitioning(int method, uint64_t tasks, uint64_t chunk, uint32_t workers, bool autoChunk){
        schedulingMethod = method;
        totalTasks = tasks;
        double tSize = (totalTasks+workers-1.0)/workers;
        mfscChunk = ceil(tSize*log(2.0)/log((1.0*tSize)));
        fissStages = getStages(totalTasks, workers);
        if(!autoChunk){    
            chunkParam = chunk;
        }
        else{
            // calculate expertChunk
            int mul = log2(totalTasks/workers)*0.618; 
            chunkParam = (totalTasks)/((2<<mul)*workers);
            method=SS;
            if (chunkParam < 1){
                chunkParam = 1;
            }
        }
        totalWorkers = workers;
        remainingTasks = tasks;
        schedulingStep = 0;
        scheduledTasks = 0;
        tssChunk = (uint64_t) ceil((double) totalTasks / ((double) 2.0*totalWorkers));
        uint64_t nTemp = (uint64_t) ceil(2.0*totalTasks/(tssChunk+1.0));
        tssDelta  = (uint64_t) (tssChunk - 1.0)/(double)(nTemp-1.0);
    }
    bool hasNextChunk(){
        return scheduledTasks < totalTasks; 
    }  
    uint64_t getNextChunk(){
        uint64_t chunkSize = 0;
        switch (schedulingMethod){
            case STATIC:{//STATIC
                chunkSize = std::ceil(totalTasks/totalWorkers);
                break;
            }
            case SS:{// self-scheduling (SS)
                chunkSize = 1;
                break;
            }
            case GSS:{//guided self-scheduling (GSS)
                chunkSize = (uint64_t) ceil((double)remainingTasks/totalWorkers);
                break;
            }
            case TSS:{//trapezoid self-scheduling (TSS)
                chunkSize = tssChunk - tssDelta * schedulingStep;
                break;
            }
            case FAC2:{//factoring (FAC2)
                uint64_t actualStep = schedulingStep/totalWorkers; // has to be an integer division 
                chunkSize = (uint64_t) ceil(pow(0.5,actualStep+1)*(totalTasks/totalWorkers));
                break;
            }
            case TFSS:{//trapezoid factoring self-scheduling (TFSS)
                chunkSize = (uint64_t) ceil((double) remainingTasks/ ((double) 2.0*totalWorkers));
                break;
            }
            case FISS:{//fixed increase self-scheduling (FISS)
                //TODO
                uint64_t X = fissStages + 2;
                uint64_t initChunk = (uint64_t) ceil(totalTasks/((2.0+fissStages)*totalWorkers));
                chunkSize = initChunk + schedulingStep * (uint64_t) ceil((2.0*totalTasks*(1.0-(fissStages/X)))/(totalWorkers*fissStages*(fissStages-1))); //chunksize with increment after init 
                break;
            }
            case VISS:{//variable increase self-scheduling (VISS)
                //TODO
                uint64_t schedulingStepnew =  schedulingStep/totalWorkers;
                uint64_t initChunk = (uint64_t) ceil(totalTasks/((2.0+fissStages)*totalWorkers));
                chunkSize =  initChunk * (uint64_t) ceil((double)(1-pow(0.5,schedulingStepnew))/0.5);
                break;
            }
            case PLS:{//performance-based loop self-scheduling (PLS)
                //TODO
                double SWR = 0.5; //static workload ratio
                if(remainingTasks > totalTasks - (totalTasks*SWR)){
                    chunkSize = (uint64_t) ceil((double)totalTasks*SWR/totalWorkers);
                }else{
                    chunkSize = (uint64_t) ceil((double)remainingTasks/totalWorkers);
                }
                break;
            }
            case PSS:{//probabilistic self-scheduling (PSS)
                //E[P] is the average number of idle processor, for now we use still totalWorkers
                double averageIdleProc = (double)totalWorkers;
                chunkSize = (uint64_t) ceil((double)remainingTasks/(1.5*averageIdleProc));
                //TODO
                break;
            }
            case MFSC:{//modifed fixed-size chunk self-scheduling (MFSC)
                chunkSize=mfscChunk;
                break;
            }
            default:{
                chunkSize = (uint64_t)ceil(totalTasks/totalWorkers/4.0);
                break;
            }
        }
        chunkSize = std::max(chunkSize,chunkParam);
        chunkSize = std::min(chunkSize, remainingTasks);
        schedulingStep++;
        scheduledTasks+=chunkSize;
        remainingTasks-=chunkSize;
        return chunkSize;
    } 
};