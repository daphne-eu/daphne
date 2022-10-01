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
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/DataPlacement.h>
#include <runtime/local/datastructures/Range.h>
#include <mpi.h>
#include <runtime/distributed/worker/MPIWorker.h>
#include <runtime/distributed/worker/MPISerializer.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>

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
template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(dctx))
    { 
        struct StoredInfo {
            size_t dp_id;
        };
        size_t messageLength=0;
        void * dataToSend;
        MPISerializer<DT>::serialize(&dataToSend, mat, isScalar, &messageLength); 
        std::vector<int> targetGroup; // We will not be able to take the advantage of broadcast if we some mpi process have the data
        int worldSize = MPIWorker::getCommSize();
        Range range;
        range.r_start = 0;
        range.c_start = 0;
        range.r_len = mat->getNumRows();
        range.c_len = mat->getNumCols();
        for (int rank=0;rank<worldSize;rank++){   
            std::string address=std::to_string(rank);  
            DataPlacement *dp = mat->getMetaDataObject().getDataPlacementByLocation(address);
            if (dp!=nullptr) {                
                mat->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);
                auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                data.ix = DistributedIndex(0, 0);
                dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);
            }
            else {  // else create new dp entry
                DistributedData data;
                data.ix = DistributedIndex(0, 0);
                AllocationDescriptorMPI allocationDescriptor (dctx, 
                                                                rank,  
                                                                data);
                dp = mat->getMetaDataObject().addDataPlacement(&allocationDescriptor, &range);
            }
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
            {
                std::cout<<"data is already placed at rank "<<rank<<std::endl;
                continue;
            }
            targetGroup.push_back(rank);
            StoredInfo storedInfo({dp->dp_id});    
        }
        if(targetGroup.size()==worldSize){
            MPIWorker::sendData(messageLength, dataToSend);
            std::cout<<"data has been send to all "<<std::endl;
        }
        else{
            for(int i=0;i<targetGroup.size();i++){
                    MPIWorker::distributeData(messageLength, dataToSend, targetGroup.at(i));
                    std::cout<<"data has been send to rank "<<targetGroup.at(i)<<std::endl;
                } 
        }
        free(dataToSend);
        long * dataAcknowledgement = (long *) malloc (sizeof(long) * 3);
        for(int i=0;i<targetGroup.size()-1;i++)
        { 
            std::cout<<"waiting for ack " << std::endl;
            int rank=MPIWorker::getDataAcknowledgement(dataAcknowledgement);
            std::cout<<"received ack form worker " << rank<<std::endl;
            std::string address=std::to_string(rank);
            DataPlacement *dp = mat->getMetaDataObject().getDataPlacementByLocation(address);
            auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
            data.identifier = dataAcknowledgement[0];
            data.numRows = dataAcknowledgement[1];
            data.numCols = dataAcknowledgement[2];
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);

        }
        free(dataAcknowledgement);
    }
};

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

        assert(mat != nullptr && "Matrix to broadcast is nullptr");
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
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix());
        }
        
        Range range;
        range.r_start = 0;
        range.c_start = 0;
        range.r_len = mat->getNumRows();
        range.c_len = mat->getNumCols();        
        for (auto i=0ul; i < workers.size(); i++){
            auto workerAddr = workers.at(i);

            // If DataPlacement dp already exists simply
            // update range (in case we have a different one) and distributed data
            DataPlacement *dp;
            if ((dp = mat->getMetaDataObject().getDataPlacementByLocation(workerAddr))) {                
                mat->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);
                auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
                data.ix = DistributedIndex(0, 0);
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
            }
            else {  // else create new dp entry
                DistributedData data;
                data.ix = DistributedIndex(0, 0);
                AllocationDescriptorGRPC allocationDescriptor (dctx, 
                                                                workerAddr,  
                                                                data);
                dp = mat->getMetaDataObject().addDataPlacement(&allocationDescriptor, &range);
            }
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            StoredInfo storedInfo({dp->dp_id});
            caller.asyncStoreCall(workerAddr, storedInfo, protoMsg);
        }       
        
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();            
            auto dp_id = response.storedInfo.dp_id;
            auto dp = mat->getMetaDataObject().getDataPlacementByID(dp_id);

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
