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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorCombine;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArgs>
struct DistributedCompute
{
    static void apply(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *combineVector, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArgs>
void distributedCompute(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *combineVector, DCTX(ctx))
{
    DistributedCompute<DTRes, DTArgs>::apply(res, numOutputs, args, numInputs, mlirCode, combineVector, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<class DTRes>
struct DistributedCompute<DTRes, const Structure>
{
    static void apply(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *combineVector,
                      DCTX(ctx))
    {
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);
        
        // For now assume that DistributedWrapper (who calls this kernel) has already allocated memory for results
        assert("result must be allocated from distributed wrapper" && res != nullptr);

        struct StoredInfo {
            std::string addr;
            DistributedIndex *ix;
        };
        DistributedCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;
        
        // Initialize Distributed index array, needed for results
        DistributedIndex *ix[numOutputs];
        for (size_t i = 0; i <numOutputs; i++)
            ix[i] = new DistributedIndex(0, 0);
        
        // Iterate over workers
        for (auto addr : workers) {                        
            distributed::Task task;
            
            // Pass all the nessecary arguments for the pipeline
            for (size_t i = 0; i < numInputs; i++) {
                auto map =  args[i]->dataPlacement.getMap();
                *task.add_inputs()->mutable_stored() = map[addr].getData();
            }
            task.set_mlir_code(mlirCode);
            StoredInfo storedInfo ({addr, nullptr});
            
            for (size_t i = 0; i < numOutputs; i++){
                storedInfo.ix = new DistributedIndex(*ix[i]);
                if (combineVector[i] == VectorCombine::ROWS)
                    ix[i] = new DistributedIndex(ix[i]->getRow() + 1, ix[i]->getCol());            
                if (combineVector[i] == VectorCombine::COLS)
                    ix[i] = new DistributedIndex(ix[i]->getRow(), ix[i]->getCol() + 1);                
            }
            // TODO for now resuing channels seems to slow things down... 
            // It is faster if we generate channel for each call and let gRPC handle resources internally
            // We might need to change this in the future and re-use channels ( data.getChannel() )
            caller.asyncComputeCall(addr, storedInfo, task);
        }

        DataPlacement::DistributedMap dataMap[numOutputs];
        // Get Results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto addr = response.storedInfo.addr;
            auto ix = response.storedInfo.ix;
            
            auto computeResult = response.result;

            for (int i = 0; i < computeResult.outputs_size(); i++){
                DistributedData data(*ix, computeResult.outputs(i).stored());                
                dataMap[i][addr] = data;
            }
        }
        for (size_t o = 0; o < numOutputs; o++){
            DataPlacement dataPlacement(dataMap[o]);
            dataPlacement.isPlacedOnWorkers = true;
            dataPlacement.combineType = combineVector[o];
            (*res[o])->dataPlacement = dataPlacement;
        }
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H