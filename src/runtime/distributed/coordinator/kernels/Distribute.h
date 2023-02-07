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
#include <runtime/local/io/DaphneSerializer.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#ifdef USE_MPI
    #include <runtime/distributed/worker/MPISerializer.h>
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
        void *dataToSend;
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
            size_t messageLength;
            MPISerializer::serializeStructure<DT>(&dataToSend, mat ,false, &messageLength, dp->range->r_start, dp->range->r_len, dp->range->c_start, dp->range->c_len);
            MPIHelper::distributeData(messageLength, dataToSend,rank);
            targetGroup.push_back(rank);
            free(dataToSend);  
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
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {
        struct StoredInfo {
            size_t dp_id;
        }; 
        
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller;
            
        assert(mat != nullptr);
        
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::DISTRIBUTE, mat, dctx);
        
        while (partioner.HasNextChunk()){ 
            auto dp = partioner.GetNextChunk();
            // Skip if already placed at workers
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            distributed::Data protoMsg;

            void *buffer = nullptr;
            size_t length;
            
            auto slicedMat = mat->sliceRow(range.r_start, range.r_start + range.r_len);
            buffer = DaphneSerializer<DT>::save(slicedMat, buffer);
            length = DaphneSerializer<DT>::length(slicedMat);
            
            protoMsg.mutable_matrix()->set_bytes(buffer, length);

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

