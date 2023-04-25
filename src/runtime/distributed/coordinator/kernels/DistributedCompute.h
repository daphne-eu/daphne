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
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/coordinator/scheduling/LoadPartitioningDistributed.h>
#ifdef USE_MPI
    #include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorCombine;



// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DTRes, class DTArgs>
struct DistributedCompute
{
    static void apply(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DTRes, class DTArgs>
void distributedCompute(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, DCTX(dctx))
{
    DistributedCompute<AT, DTRes, DTArgs>::apply(res, numOutputs, args, numInputs, mlirCode, vectorCombine, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
#ifdef USE_MPI
template<class DTRes>
struct DistributedCompute<ALLOCATION_TYPE::DIST_MPI, DTRes, const Structure>
{
    static void apply(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *vectorCombine,                      
                      DCTX(dctx))
    {
        int worldSize = MPIHelper::getCommSize() - 1; // exclude coordinator

        LoadPartitioningDistributed<DTRes, AllocationDescriptorMPI>::SetOutputsMetadata(res, numOutputs, vectorCombine, dctx);
        
        std::vector<char> taskBuffer;        
        void *taskToSend;
        size_t messageLengths[worldSize]; 
        for (int rank=0;rank<worldSize;rank++) // we currently exclude the coordinator
        {

            distributed::Task task;
            std::string addr= std::to_string(rank+1);
            for (size_t i = 0; i < numOutputs; i++)
            {
                auto dp = args[i]->getMetaDataObject()->getDataPlacementByLocation(addr);
                auto distrData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                distributed::StoredData protoData;
                //std::cout<<"identifier " << distrData.identifier<<std::endl;
                protoData.set_identifier(distrData.identifier); 
                protoData.set_num_cols(distrData.numCols);
                protoData.set_num_rows(distrData.numRows);

                *task.add_inputs()->mutable_stored() = protoData;
            }
            task.set_mlir_code(mlirCode);
            MPISerializer::serializeTask(&taskToSend, &messageLengths[rank], &task);
            MPIHelper::distributeTask(messageLengths[rank], taskToSend,rank+1);
            free(taskToSend);
        }

    }
};
#endif

// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------


template<class DTRes>
struct DistributedCompute<ALLOCATION_TYPE::DIST_GRPC, DTRes, const Structure>
{
    static void apply(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *vectorCombine,                      
                      DCTX(dctx))
    {
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
        
        struct StoredInfo {
            std::string addr;
        };                
        DistributedGRPCCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;
        
        // Set output meta data
        LoadPartitioningDistributed<DTRes, AllocationDescriptorGRPC>::SetOutputsMetadata(res, numOutputs, vectorCombine, dctx);
        
        // Iterate over workers
        // Pass all the nessecary arguments for the pipeline
        for (auto addr : workers) {

            distributed::Task task;
            for (size_t i = 0; i < numInputs; i++){
                auto dp = args[i]->getMetaDataObject()->getDataPlacementByLocation(addr);
                auto distrData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();

                distributed::StoredData protoData;
                protoData.set_identifier(distrData.identifier);
                protoData.set_num_cols(distrData.numCols);
                protoData.set_num_rows(distrData.numRows);

                *task.add_inputs()->mutable_stored() = protoData;
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
            
            for (int o = 0; o < computeResult.outputs_size(); o++){            
                auto resMat = *res[o];
                auto dp = resMat->getMetaDataObject()->getDataPlacementByLocation(addr);

                auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
                data.identifier = computeResult.outputs()[o].stored().identifier();
                data.numRows = computeResult.outputs()[o].stored().num_rows();
                data.numCols = computeResult.outputs()[o].stored().num_cols();
                data.isPlacedAtWorker = true;
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);                                                
            }            
        }                
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H
