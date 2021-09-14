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

#ifndef SRC_RUNTIME_LOCAL_VECTORIZED_LOADPARTITIONING_H
#define SRC_RUNTIME_LOCAL_VECTORIZED_LOADPARTITIONING_H

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
public:
    LoadPartitioning(int method, uint64_t tasks, uint64_t chunk, uint32_t workers){ 
        schedulingMethod = method;
        totalTasks = tasks;
        chunkParam = chunk;
        totalWorkers = workers;
        remainingTasks = tasks;
        schedulingStep = 0;
        scheduledTasks = 0;
        tssChunk = (uint64_t) ceil((double) totalTasks / ((double) 2*totalWorkers));
        uint64_t nTemp = (uint64_t) ceil(2*totalTasks/(tssChunk+1));
        tssDelta  = (uint64_t) (tssChunk - 1)/(double)(nTemp-1);
    }
    bool hasNextChunk(){
        return scheduledTasks < totalTasks; 
    }  
    uint64_t getNextChunk(){
        uint64_t chunkSize = 0;
        switch (schedulingMethod){
            case 0:{//STATIC
                    chunkSize = (uint64_t)ceil(totalTasks/totalWorkers);
                    break;
            }
            case 1:{// SS
                    chunkSize = 1;
                    break;
            }
            case 2:{//GSS
                    chunkSize = (uint64_t)ceil((double)remainingTasks/totalTasks);
                    break;
            }
            case 3:{//TSS
                    chunkSize = tssChunk - tssDelta * schedulingStep;
                    break;
            }
            case 4:{//FAC2
                    uint64_t actualStep = schedulingStep/totalWorkers; // has to be an integer division 
                    chunkSize = (uint64_t) ceil(pow(0.5,actualStep+1)*(totalTasks/totalWorkers));
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

#endif //SRC_RUNTIME_LOCAL_VECTORIZED_LOADPARTITIONING_H
