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
#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <runtime/local/datastructures/DistributedAllocationHelpers.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/local/datastructures/DataPlacement.h>
#include <runtime/local/datastructures/Range.h>
#include <runtime/distributed/worker/WorkerImpl.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/coordinator/scheduling/LoadPartitioningDistributed.h>

#ifdef USE_MPI
#include <runtime/distributed/worker/MPIHelper.h>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/distributed/worker/MPIWorker.h>
#include <runtime/distributed/worker/MPISerializer.h>
#endif

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct Broadcast {
    static void apply(DT *mat, bool isScalar, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void broadcast(DT *&mat, bool isScalar, DCTX(dctx))
{
    Broadcast<AT, DT>::apply(mat, isScalar, dctx);
}


// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
#ifdef USE_MPI
template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(dctx))
    {
        size_t messageLength=0;
        void * dataToSend;
        //auto ptr = (double*)(&mat);
        MPISerializer::serializeStructure<DT>(&dataToSend, mat, isScalar, &messageLength); 
        std::vector<int> targetGroup; // We will not be able to take the advantage of broadcast if some mpi processes have the data
        
        LoadPartitioningDistributed<DT, AllocationDescriptorMPI> partioner(DistributionSchema::BROADCAST, mat, dctx);
        while (partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            auto rank = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getRank();
            
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
            {
                //std::cout<<"data is already placed at rank "<<rank<<std::endl;
                auto data =dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                MPIHelper::sendObjectIdentifier(data.identifier, rank);
               // std::cout<<"Identifier ( "<<data.identifier<< " ) has been send to " <<(rank+1)<<std::endl;
                continue;
            }
            targetGroup.push_back(rank);  
        }
        if((int)targetGroup.size()==MPIHelper::getCommSize() - 1){ // exclude coordinator
            MPIHelper::sendData(messageLength, dataToSend);
           // std::cout<<"data has been send to all "<<std::endl;
        }
        else{
            for(int i=0;i<(int)targetGroup.size();i++){
                    MPIHelper::distributeData(messageLength, dataToSend, targetGroup.at(i));
                    //std::cout<<"data has been send to rank "<<targetGroup.at(i)<<std::endl;
                } 
        }
        free(dataToSend);
        for(int i=0;i<(int)targetGroup.size();i++)
        { 
            //std::cout<<"From broadcast waiting for ack " << std::endl;
           
            int rank=targetGroup.at(i);
            if (rank==COORDINATOR)
            {

               // std::cout<<"coordinator doe not need ack from itself" << std::endl;
                continue;
            }
            WorkerImpl::StoredInfo dataAcknowledgement=MPIHelper::getDataAcknowledgement(&rank);
            //std::cout<<"received ack form worker " << rank<<std::endl;
            std::string address=std::to_string(rank);
            DataPlacement *dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);
            auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
            data.identifier = dataAcknowledgement.identifier;
            data.numRows = dataAcknowledgement.numRows;
            data.numCols = dataAcknowledgement.numCols;
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);

        }
    }
};
#endif
// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(dctx)) 
    {
        struct StoredInfo {
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller;
        
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
        
        distributed::Data protoMsg;

        double *val;
        if (isScalar) {
            auto ptr = (double*)(&mat);
            val = ptr;
            auto protoVal = protoMsg.mutable_value();
            protoVal->set_f64(*val);
            // Need matrix for metadata, type of matrix does not really matter.
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
        } 
        else { // Not scalar
            assert(mat != nullptr && "Matrix to broadcast is nullptr");
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix());
        }
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::BROADCAST, mat, dctx);
        
        while(partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            StoredInfo storedInfo({dp->dp_id});
            caller.asyncStoreCall(dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation(), storedInfo, protoMsg);
        }       
        
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();            
            auto dp_id = response.storedInfo.dp_id;
            auto dp = mat->getMetaDataObject()->getDataPlacementByID(dp_id);

            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();

            auto storedData = response.result;
            data.identifier = storedData.identifier();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;

            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);            
        }                
    };           
};
