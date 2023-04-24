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
        int worldSize= MPIHelper::getCommSize()-1; // exclude coordinator
        // Initialize Distributed index array, needed for results
        for (size_t i = 0; i < numOutputs; i++)
        {
            size_t partitionSize=0, remainingSize=0, rowCount=0,colCount=0; // startRow=0, startCol=0;
            auto combineType = vectorCombine[i];
            remainingSize = (combineType==VectorCombine::ROWS)? (*res[i])->getNumRows(): (*res[i])->getNumCols();
            partitionSize = (combineType==VectorCombine::ROWS)? (*res[i])->getNumRows()/worldSize: (*res[i])->getNumCols()/worldSize;
            if(partitionSize<1)
            {
                partitionSize = 1;
                worldSize= (combineType==VectorCombine::ROWS)? (*res[i])->getNumRows() : (*res[i])->getNumCols();
            }
            for(int rank=0; rank<worldSize;rank++) // we currently exclude the coordinator
            {      
                DistributedData data;
                data.vectorCombine = combineType;
                data.isPlacedAtWorker = true;
                Range range;
                if(rank==worldSize-1 ){
                    rowCount= remainingSize;
                    colCount = remainingSize;
                }
                else  {
                    rowCount=partitionSize;
                    colCount=partitionSize;
                }                                
                if (combineType== VectorCombine::ROWS) {
                    data.ix  = DistributedIndex(rank, 0);              
                    colCount=(*res[i])->getNumCols();       
                    range.r_start = data.ix.getRow() * partitionSize;
                    range.r_len = rowCount;
                    range.c_start = 0;
                    range.c_len = colCount;
                }
                if (vectorCombine[i] == VectorCombine::COLS) {
                    data.ix  = DistributedIndex(0, rank);  
                    rowCount= (*res[i])->getNumRows();
                    range.r_start = 0; 
                    range.r_len = rowCount; 
                    range.c_start = data.ix.getCol() * partitionSize;
                    range.c_len = colCount;
                }
               // std::cout<<"rank "<< rank+1 <<" Range rows from "<< range.r_start <<" to " <<( range.r_len + range.r_start)<< " cols from " <<range.c_start <<" to " <<( range.c_len + range.c_start)<<std::endl;
                remainingSize-=partitionSize;
                std::string addr= std::to_string(rank+1);
                // If dp already exists for this worker, update the range and data
                if (auto dp = (*res[i])->getMetaDataObject()->getDataPlacementByLocation(addr)) { 
                    (*res[i])->getMetaDataObject()->updateRangeDataPlacementByID(dp->dp_id, &range);
                    dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);                    
                }
                else { // else create new dp entry   
                    AllocationDescriptorMPI allocationDescriptor(
                                            rank+1,
                                            dctx,
                                            data);                                    
                    ((*res[i]))->getMetaDataObject()->addDataPlacement(&allocationDescriptor, &range);                    
                } 
            }
        }
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

        // Initialize Distributed index array, needed for results
        std::vector<DistributedIndex> ix(numOutputs, DistributedIndex(0, 0));
        
        // Iterate over workers
        // Pass all the nessecary arguments for the pipeline
        for (auto addr : workers) {
            // Set output meta data
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

                DistributedData data;
                data.ix = ix[i];
                data.vectorCombine = vectorCombine[i];
                data.isPlacedAtWorker = true;
                                
                // Update distributed index for next iteration
                // and set ranges for objmetadata
                Range range;
                if (vectorCombine[i] == VectorCombine::ROWS) {
                    ix[i] = DistributedIndex(ix[i].getRow() + 1, ix[i].getCol());            
                    
                    range.r_start = data.ix.getRow() * k + std::min(data.ix.getRow(), m);
                    range.r_len = ((data.ix.getRow() + 1) * k + std::min((data.ix.getRow() + 1), m)) - range.r_start;
                    range.c_start = 0;
                    range.c_len = (*res[i])->getNumCols();
                }
                if (vectorCombine[i] == VectorCombine::COLS) {
                    ix[i] = DistributedIndex(ix[i].getRow(), ix[i].getCol() + 1);
                    
                    range.r_start = 0; 
                    range.r_len = (*res[i])->getNumRows(); 
                    range.c_start = data.ix.getCol() * k + std::min(data.ix.getCol(), m);
                    range.c_len = ((data.ix.getCol() + 1) * k + std::min((data.ix.getCol() + 1), m)) - range.c_start;
                }

                // If dp already exists for this worker, update the range and data
                if (auto dp = (*res[i])->getMetaDataObject()->getDataPlacementByLocation(addr)) { 
                    (*res[i])->getMetaDataObject()->updateRangeDataPlacementByID(dp->dp_id, &range);
                    dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);                    
                }
                else { // else create new dp entry   
                    AllocationDescriptorGRPC allocationDescriptor(
                                            dctx,
                                            addr,
                                            data);                                    
                    ((*res[i]))->getMetaDataObject()->addDataPlacement(&allocationDescriptor, &range);                    
                } 
            }

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
