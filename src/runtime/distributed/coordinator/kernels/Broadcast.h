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
        double val = 1;
        if (isScalar){
            auto ptr = (double*)(&mat);
            val = *ptr;
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false);
        }
        std::vector<int> targetGroup; // We will not be able to take the advantage of broadcast if some mpi processes have the data
        
        LoadPartitioningDistributed<DT, AllocationDescriptorMPI> partioner(DistributionSchema::BROADCAST, mat, dctx);
        while (partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            auto rank = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getRank();
            
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            // Minimum chunk size
            auto min_chunk_size = dctx->config.max_distributed_serialization_chunk_size < DaphneSerializer<DT>::length(mat) ? 
                        dctx->config.max_distributed_serialization_chunk_size : 
                        DaphneSerializer<DT>::length(mat);

            MPIHelper::initiateStreaming(rank, min_chunk_size);
            targetGroup.push_back(rank);  
        }

        if((int)targetGroup.size()==MPIHelper::getCommSize() - 1){ // exclude coordinator
            if (isScalar){
                std::vector<char> buffer;
                auto length = DaphneSerializer<double>::serialize(val, buffer);
                MPIHelper::broadcastData(length, buffer.data());
            } else {
                auto serializer = DaphneSerializerChunks<DT>(mat, dctx->config.max_distributed_serialization_chunk_size);
                for (auto it = serializer.begin(); it != serializer.end(); ++it)
                    MPIHelper::broadcastData(it->first, it->second->data());
            }
        }
        else{
            for(int i=0;i<(int)targetGroup.size();i++)
                MPIHelper::sendData(messageLength, dataToSend.data(), targetGroup.at(i));
        }
        for(int i=0;i<(int)targetGroup.size();i++)
        {            
            int rank = targetGroup.at(i);
            if (rank == COORDINATOR)
                continue;

            WorkerImpl::StoredInfo dataAcknowledgement = MPIHelper::getDataAcknowledgement(&rank);
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
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(dctx)) 
    {
        struct StoredInfo {
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller(dctx);
        
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
        
        distributed::Data protoMsg;
        double val = 1;
        if (isScalar) {
            auto ptr = (double*)(&mat);        
            val = *ptr;
            // Need matrix for metadata, type of matrix does not really matter.
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
        }         
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::BROADCAST, mat, dctx);
        
        while(partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            auto address = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation();
            
            StoredInfo storedInfo({dp->dp_id});
            caller.asyncStoreCall(address, storedInfo);
            // Minimum chunk size
            auto min_chunk_size = dctx->config.max_distributed_serialization_chunk_size < DaphneSerializer<DT>::length(mat) ? 
                        dctx->config.max_distributed_serialization_chunk_size : 
                        DaphneSerializer<DT>::length(mat);

            // First send chunk size
            protoMsg.set_bytes(&min_chunk_size, sizeof(size_t));
            caller.sendDataStream(address, protoMsg);
            if (isScalar) {
                std::vector<char> buffer;
                auto length = DaphneSerializer<double>::serialize(val, buffer);
                protoMsg.set_bytes(buffer.data(), length);

                caller.sendDataStream(address, protoMsg);
            } else{
                auto serializer = DaphneSerializerChunks<DT>(mat, min_chunk_size);
                for (auto it = serializer.begin(); it != serializer.end(); ++it){                
                    protoMsg.set_bytes(it->second->data(), it->first);
                    caller.sendDataStream(address, protoMsg);
                }
            }
        }
        caller.writesDone();

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

// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(dctx)) 
    {
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
    
        std::vector<std::thread> threads_vector;
        std::vector<char> buffer;
        double val = 1;
        if (isScalar) {
            auto ptr = (double*)(&mat);        
            val = *ptr;            
            // Need matrix for metadata, type of matrix does not really matter.
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
        } 
        LoadPartitioningDistributed<DT, AllocationDescriptorGRPC> partioner(DistributionSchema::BROADCAST, mat, dctx);

        while(partioner.HasNextChunk()){
            auto dp = partioner.GetNextChunk();
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            auto workerAddr = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation();
            std::thread t([=, &mat]() 
            {
                // TODO Consider saving channels inside DaphneContext
                grpc::ChannelArguments ch_args;
                ch_args.SetMaxSendMessageSize(-1);
                ch_args.SetMaxReceiveMessageSize(-1);
                auto channel = grpc::CreateCustomChannel(workerAddr, grpc::InsecureChannelCredentials(), ch_args);
                auto stub = distributed::Worker::NewStub(channel);
                distributed::StoredData storedData;
                grpc::ClientContext grpc_ctx;
                auto writer = stub->Store(&grpc_ctx, &storedData);
                distributed::Data protoMsg;
                
                if (isScalar){
                    std::vector<char> buffer;
                    auto length = DaphneSerializer<double>::serialize(val, buffer);
                    protoMsg.set_bytes(buffer.data(), length);

                    writer->Write(protoMsg);
                } else {
                    auto serializer = DaphneSerializerChunks<DT>(mat, dctx->config.max_distributed_serialization_chunk_size);
                    for (auto it = serializer.begin(); it != serializer.end(); ++it){                
                        protoMsg.set_bytes(it->second->data(), it->first);
                        writer->Write(protoMsg);
                    }  
                }
                writer->WritesDone();
                auto status = writer->Finish();
                
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
    };           
};
