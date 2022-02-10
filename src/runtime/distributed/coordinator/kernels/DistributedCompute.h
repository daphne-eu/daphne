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

#include <grpcpp/grpcpp.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorCombine;


// ****************************************************************************
// Template speciliazations for each communication framework (gRPC, OPENMPI, etc.)
// ****************************************************************************

template<DISTRIBUTED_BACKEND backend, class DTRes>
struct ComputeImplementation
{
    static void apply(DTRes **&res, size_t numOutputs, const Structure **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, DCTX(ctx)) = delete;    
};

template<DISTRIBUTED_BACKEND backend, class DTRes>
void computeImplementation(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *vectorCombine,
                      DCTX(ctx))
{
    ComputeImplementation<backend, DTRes>::apply(res, numOutputs, args, numInputs, mlirCode, vectorCombine, ctx);
}
// ****************************************************************************
// gRPC implementation
// ****************************************************************************
template<class DTRes>
struct ComputeImplementation<DISTRIBUTED_BACKEND::GRPC, DTRes>
{    
    static void apply(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *vectorCombine,
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
        struct StoredInfo {
            std::string addr;
        };                
        DistributedCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;

        // Broadcast at workers
        for (auto addr : workers) {                        
            distributed::Task task;
            
            // Pass all the nessecary arguments for the pipeline
            for (size_t i = 0; i < numInputs; i++) {
                // Find input for this worker
                auto omd = args[i]->getObjectMetaDataByLocation(addr);

                auto distrData = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
                distributed::StoredData protoData;
                protoData.set_filename(distrData.filename);
                protoData.set_num_rows(distrData.numRows);
                protoData.set_num_cols(distrData.numCols); 
                
                *task.add_inputs()->mutable_stored()= protoData;
            }
            task.set_mlir_code(mlirCode);           
            StoredInfo storedInfo({addr});    
            // TODO for now resuing channels seems to slow things down... 
            // It is faster if we generate channel for each call and let gRPC handle resources internally
            // We might need to change this in the future and re-use channels ( data.getChannel() )
            caller.asyncComputeCall(addr, storedInfo, task);
        }

        
        // Get Results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto addr = response.storedInfo.addr;
            
            auto computeResult = response.result;            
            
            for (int i = 0; i < computeResult.outputs_size(); i++){
                auto omd = (*res[i])->getObjectMetaDataByLocation(addr);                
                DistributedData data = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();

                data.filename = computeResult.outputs()[i].stored().filename();
                data.numRows = computeResult.outputs()[i].stored().num_rows();
                data.numCols = computeResult.outputs()[i].stored().num_cols();
                dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);                  
            }            
        }
    };
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArgs>
struct DistributedCompute
{
    static void apply(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArgs>
void distributedCompute(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, DCTX(ctx))
{
    DistributedCompute<DTRes, DTArgs>::apply(res, numOutputs, args, numInputs, mlirCode, vectorCombine, ctx);
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
                      VectorCombine *vectorCombine,
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
        

        // Initialize Distributed index array, needed for results
        DistributedIndex *ix[numOutputs];
        for (size_t i = 0; i <numOutputs; i++)
            ix[i] = new DistributedIndex(0, 0);
        

        
        
         // Iterate over workers
        // Pass all the nessecary arguments for the pipeline
        for (auto addr : workers) {
            for (size_t i = 0; i < numOutputs; i++){                 
                // Get Result ranges
                auto combineType = vectorCombine[i];
                auto workersSize = workers.size();
                size_t k = 0, m = 0;                
                if (combineType == VectorCombine::ROWS) {
                    k = (*res[i])->getNumRows() / workersSize;
                    m = (*res[i])->getNumRows() % workersSize;
                }
                else if (combineType == VectorCombine::COLS){
                    k = (*res[i])->getNumCols() / workersSize;
                    m = (*res[i])->getNumCols() % workersSize;
                }
                else
                    assert(!"Only Rows/Cols combineType supported atm");

                IAllocationDescriptor *allocationDescriptor;
                DistributedData data;
                data.ix = *ix[i];
                data.vectorCombine = vectorCombine[i];
                data.isPlacedAtWorker = true;

                allocationDescriptor = new AllocationDescriptorDistributed(
                                                ctx,
                                                addr,
                                                data);
                // Update distributed index for next iteration
                // and set ranges for objmetadata
                Range range;
                if (vectorCombine[i] == VectorCombine::ROWS) {
                    ix[i] = new DistributedIndex(ix[i]->getRow() + 1, ix[i]->getCol());            
                    
                    range.r_start = data.ix.getRow() * k + std::min(data.ix.getRow(), m);
                    range.r_len = ((data.ix.getRow() + 1) * k + std::min((data.ix.getRow() + 1), m)) - range.r_start;
                    range.c_start = 0;
                    range.c_len = (*res[i])->getNumCols();
                }
                if (vectorCombine[i] == VectorCombine::COLS) {
                    ix[i] = new DistributedIndex(ix[i]->getRow(), ix[i]->getCol() + 1);
                    
                    range.r_start = 0; 
                    range.r_len = (*res[i])->getNumRows(); 
                    range.c_start = data.ix.getCol() * k + std::min(data.ix.getCol(), m);
                    range.c_len = ((data.ix.getCol() + 1) * k + std::min((data.ix.getCol() + 1), m)) - range.c_start;
                }

                // If omd already exists for this worker, update the range and data
                if (auto omd = (*res[i])->getObjectMetaDataByLocation(addr)) { 
                    (*res[i])->updateRangeObjectMetaDataByID(omd->omd_id, &range);
                    dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);                    
                }
                else { // else create new omd entry
                    allocationDescriptor = new AllocationDescriptorDistributed(
                                                ctx, 
                                                addr,
                                                data);            
                    ((*res[i]))->addObjectMetaData(allocationDescriptor, &range);                    
                } 
            }
        }

        // TODO choose implementation
        // Possible to call all implementations and each one will try to find allocators (MPI, gRPC)
        // if none is find, then the implementation simply returns.
        computeImplementation<DISTRIBUTED_BACKEND::GRPC, DTRes>(res, numOutputs, args, numInputs, mlirCode, vectorCombine, ctx);                
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H