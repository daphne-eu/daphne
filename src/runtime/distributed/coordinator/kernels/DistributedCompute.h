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
        size_t worldSize = MPIHelper::getCommSize(); // exclude coordinator

        LoadPartitioningDistributed<DTRes, AllocationDescriptorMPI>::SetOutputsMetadata(res, numOutputs, vectorCombine, dctx);
        
        std::vector<char> taskBuffer;
        for (size_t rank = 1; rank < worldSize; rank++) // we currently exclude the coordinator
        {
            MPIHelper::Task task;
            std::string addr= std::to_string(rank);
            for (size_t i = 0; i < numInputs; i++)
            {
                auto dp = args[i]->getMetaDataObject()->getDataPlacementByLocation(addr);
                auto distrData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                
                MPIHelper::StoredInfo storedData({distrData.identifier, distrData.numRows, distrData.numCols});                
                task.inputs.push_back(storedData);
            }
            task.mlir_code = mlirCode;
            task.serialize(taskBuffer);
            auto len = task.sizeInBytes();
            MPIHelper::sendTask(len, taskBuffer.data(), rank);
        }

        for (size_t rank = 1; rank < worldSize; rank++){
            auto buffer = MPIHelper::getComputeResults(rank);
            std::vector<WorkerImpl::StoredInfo> infoVec = MPIHelper::constructStoredInfoVector(buffer);
            size_t idx = 0;
            for (auto info : infoVec){            
                auto resMat = *res[idx++];
                auto dp = resMat->getMetaDataObject()->getDataPlacementByLocation(std::to_string(rank));

                auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                data.identifier = info.identifier;
                data.numRows = info.numRows;
                data.numCols = info.numCols;
                data.isPlacedAtWorker = true;
                dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);                                                
            }
        }
    }
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------


template<class DTRes>
struct DistributedCompute<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DTRes, const Structure>
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
        DistributedGRPCCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller(dctx);
        
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


// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------


template<class DTRes>
struct DistributedCompute<ALLOCATION_TYPE::DIST_GRPC_SYNC, DTRes, const Structure>
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
        
        // Initialize Distributed index array, needed for results
        std::vector<DistributedIndex> ix(numOutputs, DistributedIndex(0, 0));
        
        std::vector<std::thread> threads_vector;
        
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
            std::thread t([&, task, addr]()
            {
                auto stub = ctx->stubs[addr].get();

                distributed::ComputeResult computeResult;
                grpc::ClientContext grpc_ctx;

                auto status = stub->Compute(&grpc_ctx, task, &computeResult);
                if (!status.ok())
                    throw std::runtime_error(status.error_message());

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
            });
            threads_vector.push_back(move(t));           
        }
        for (auto &thread : threads_vector)
            thread.join();
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H
