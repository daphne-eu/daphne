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
#include <runtime/local/io/DaphneSerializer.h>

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
        std::vector<char> dataToSend;
        if (isScalar){
            auto ptr = (double*)(&mat);
            double val = *ptr;
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false);
            dataToSend.reserve(sizeof(double));
            messageLength = DaphneSerializer<double>::serialize(val, dataToSend);        
        }
        else {
            messageLength = DaphneSerializer<typename std::remove_const<DT>::type>::serialize(mat, dataToSend);
        }
        std::vector<int> targetGroup; // We will not be able to take the advantage of broadcast if some mpi processes have the data
        
        LoadPartitioningDistributed<DT, AllocationDescriptorMPI> partioner(DistributionSchema::BROADCAST, mat, dctx);
        while (partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            auto rank = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getRank();
            
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
            {
                //std::cout<<"data is already placed at rank "<<rank<<std::endl;
                auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                MPIHelper::sendObjectIdentifier(data.identifier, rank);
               // std::cout<<"Identifier ( "<<data.identifier<< " ) has been send to " <<(rank+1)<<std::endl;
                continue;
            }
            targetGroup.push_back(rank);  
        }
        if((int)targetGroup.size()==MPIHelper::getCommSize() - 1){ // exclude coordinator
            MPIHelper::sendData(messageLength, dataToSend.data());
           // std::cout<<"data has been send to all "<<std::endl;
        }
        else{
            for(int i=0;i<(int)targetGroup.size();i++){
                    MPIHelper::distributeData(messageLength, dataToSend.data(), targetGroup.at(i));
                    //std::cout<<"data has been send to rank "<<targetGroup.at(i)<<std::endl;
                } 
        }
        for(int i=0;i<(int)targetGroup.size();i++)
        { 
            //std::cout<<"From broadcast waiting for ack " << std::endl;
           
            int rank = targetGroup.at(i);
            if (rank == COORDINATOR)
            {

               // std::cout<<"coordinator doe not need ack from itself" << std::endl;
                continue;
            }
            WorkerImpl::StoredInfo dataAcknowledgement = MPIHelper::getDataAcknowledgement(&rank);
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

        std::vector<char> buffer;
        if (isScalar) {
            auto ptr = (double*)(&mat);        
            double val = *ptr;
            auto length = DaphneSerializer<double>::serialize(val, buffer);
            protoMsg.set_bytes(buffer.data(), length);

            // Need matrix for metadata, type of matrix does not really matter.
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
        } 
        else { // Not scalar
            // DT is const Structure, but we only provide template specialization for structure.
            // TODO should we implement an additional specialization or remove constness from template parameter?
            size_t length = DaphneSerializer<typename std::remove_const<DT>::type>::serialize(mat, buffer);
            protoMsg.set_bytes(buffer.data(), length);            
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
