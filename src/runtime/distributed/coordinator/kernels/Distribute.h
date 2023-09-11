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

#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/coordinator/scheduling/LoadPartitioningDistributed.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#ifdef USE_MPI
    #include <runtime/distributed/worker/MPIHelper.h>
#endif 

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct Distribute {
    static void apply(DT *mat, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distribute(DT *mat, DCTX(dctx))
{
    Distribute<AT, DT>::apply(mat, dctx);
}


// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

#ifdef USE_MPI
// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {        
        std::vector<char> dataToSend;
        std::vector<int> targetGroup;  

        LoadPartitioningDistributed<DT, AllocationDescriptorMPI> partioner(DistributionSchema::DISTRIBUTE, mat, dctx);        
        
        while (partioner.HasNextChunk()){
            DataPlacement *dp = partioner.GetNextChunk();
            auto rank = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getRank();
            
            //std::cout<<"rank "<< rank+1<< " will work on rows from " << startRow << " to "  << startRow+rowCount<<std::endl;
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
            {
               // std::cout<<"worker already has the data"<<std::endl;
               auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
               MPIHelper::sendObjectIdentifier(data.identifier, rank);
               //std::cout<<"Identifier ( "<<data.identifier<< " ) has been send to " <<(rank+1)<<std::endl;
               continue;
            }
            auto slicedMat = mat->sliceRow(dp->range->r_start, dp->range->r_start + dp->range->r_len);
            auto len = DaphneSerializer<typename std::remove_const<DT>::type>::serialize(slicedMat, dataToSend);                        
            MPIHelper::distributeData(len, dataToSend.data(),rank);
            targetGroup.push_back(rank);            
        }
        for(size_t i=0;i<targetGroup.size();i++)
        {
            int rank=targetGroup.at(i);
            //std::cout<<"From distribute waiting for ack ("+std::to_string(rank)+")" << std::endl;
            if (rank==COORDINATOR)
            {

               // std::cout<<"coordinator doe not need ack from itself" << std::endl;
                continue;
            }
            WorkerImpl::StoredInfo dataAcknowledgement = MPIHelper::getDataAcknowledgement(&rank);
            std::string address = std::to_string(rank);
            DataPlacement *dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);
            auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
            data.identifier = dataAcknowledgement.identifier ;
            data.numRows = dataAcknowledgement.numRows;
            data.numCols = dataAcknowledgement.numCols;
            data.isPlacedAtWorker = true;
            //std::cout<<"acknowledgement received with distribute identifier " << data.identifier<<std::endl;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);
        }

    }
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {
        struct StoredInfo {
            size_t dp_id;
        }; 
        
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller(dctx);
            
        assert(mat != nullptr);
        
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::DISTRIBUTE, mat, dctx);
        
        while (partioner.HasNextChunk()){ 
            auto dp = partioner.GetNextChunk();
            // Skip if already placed at workers
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            distributed::Data protoMsg;

            std::vector<char> buffer;
            
            auto slicedMat = mat->sliceRow(dp->range->r_start, dp->range->r_start + dp->range->r_len);
            // DT is const Structure, but we only provide template specialization for structure.
            // TODO should we implement an additional specialization or remove constness from template parameter?
            auto length = DaphneSerializer<typename std::remove_const<DT>::type>::serialize(slicedMat, buffer);            
            protoMsg.set_bytes(buffer.data(), length);

            StoredInfo storedInfo({dp->dp_id}); 
            caller.asyncStoreCall(dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation(), storedInfo, protoMsg);
        }                
                       

        // get results       
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            
            auto storedData = response.result;            

            auto dp = mat->getMetaDataObject()->getDataPlacementByID(dp_id);
            
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
            data.identifier = storedData.identifier();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);            
        }

    }
};

// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
        
        assert(mat != nullptr);
        
        std::vector<std::thread> threads_vector;
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::DISTRIBUTE, mat, dctx);
        while (partioner.HasNextChunk()){ 
            auto dp = partioner.GetNextChunk();
            // Skip if already placed at workers
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            distributed::Data protoMsg;

            std::vector<char> buffer;
            
            auto slicedMat = mat->sliceRow(dp->range->r_start, dp->range->r_start + dp->range->r_len);
            // DT is const Structure, but we only provide template specialization for structure.
            // TODO should we implement an additional specialization or remove constness from template parameter?
            auto length = DaphneSerializer<typename std::remove_const<DT>::type>::serialize(slicedMat, buffer);            
            protoMsg.set_bytes(buffer.data(), length);

            auto workerAddr = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation();
            std::thread t([=]()
            {
                auto stub = ctx->stubs[workerAddr].get();

                distributed::StoredData storedData;
                grpc::ClientContext grpc_ctx;
                auto status = stub->Store(&grpc_ctx, protoMsg, &storedData);
                if (!status.ok())
                    throw std::runtime_error(status.error_message());

                DistributedData newData;
                newData.identifier = storedData.identifier();
                newData.numRows = storedData.num_rows();
                newData.numCols = storedData.num_cols();
                newData.isPlacedAtWorker = true;
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(newData);
            });
            threads_vector.push_back(move(t));            
        }
        for (auto &thread : threads_vector)
            thread.join();
    }
};
