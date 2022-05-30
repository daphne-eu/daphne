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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H

#include <ir/daphneir/Daphne.h>

#include <runtime/distributed/coordinator/kernels/Broadcast.h>
#include <runtime/distributed/coordinator/kernels/Distribute.h>
#include <runtime/distributed/coordinator/kernels/DistributedCollect.h>
#include <runtime/distributed/coordinator/kernels/DistributedCompute.h>

#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>
using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

template <class DT>
class DistributedWrapper {
private:
    DCTX(_ctx);

protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, const Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }
public:
    DistributedWrapper(DCTX(ctx)) : _ctx(ctx) {
        //TODO start workers from here instead of manually (e.g. resource manager) ? 

    }
    ~DistributedWrapper() = default; //TODO Terminate workers (e.g. with gRPC, resource manager, etc.)

    void execute(const char *mlirCode,
                 DT ***res,
                 const Structure **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 int64_t *outRows,
                 int64_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines)                 
    {        
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        // assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);        
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);
        
        // Backend Implementation 
        // gRPC hard-coded selection
        // TODO choose implementation based on configFile/command-line argument        
        auto alloc_type = ALLOCATION_TYPE::DIST_GRPC;

        // output allocation for row-wise combine
        for(size_t i = 0; i < numOutputs; ++i) {
            if(*(res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                // TODO we know result is only DenseMatrix<double> for now,
                // but in the future this will change to support other DataTypes
                *(res[i]) = DataObjectFactory::create<DT>(outRows[i], outCols[i], zeroOut);
            }
        }
        
        // Distribute and broadcast inputs        
        // Each primitive sends information to workers and changes the Structures' metadata information 
        for (auto i = 0u; i < numInputs; ++i) {
            // if already placed on workers, skip
            // TODO maybe this is not enough. We might also need to check if data resides in the specific way we need to.
            // (i.e. rows/cols splitted accordingly). If it does then we can skip.
                    
            if (isBroadcast(splits[i], inputs[i])){
                broadcast(inputs[i], alloc_type, _ctx);
            }
            else {
                distribute(inputs[i], alloc_type, _ctx);
            }
        }
          
        distributedCompute(res, numOutputs, inputs, numInputs, mlirCode, combines, alloc_type, _ctx);

        // Collect
        for (size_t o = 0; o < numOutputs; o++){
            assert ((combines[o] == VectorCombine::ROWS || combines[o] == VectorCombine::COLS) && "we only support rows/cols combine atm");
            distributedCollect(*res[o], alloc_type, _ctx);           
        }
        
        
    }
};


#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
