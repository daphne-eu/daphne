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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/local/io/DaphneSerializer.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#ifdef USE_MPI
    #include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cassert>
#include <cstddef>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct DistributedCollect {
    static void apply(DT *&mat, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distributedCollect(DT *&mat, DCTX(dctx))
{
    DistributedCollect<AT, DT>::apply(mat, dctx);
}



// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************


// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
#ifdef USE_MPI
template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        
        size_t worldSize = MPIHelper::getCommSize();
        for(size_t rank=0; rank<worldSize ; rank++) 
        {
            if(rank==COORDINATOR) // we currently exclude the coordinator
               continue;
            
            std::string address = std::to_string(rank);  
            auto dp=mat->getMetaDataObject()->getDataPlacementByLocation(address);   
            auto distributedData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();            
            WorkerImpl::StoredInfo info = {
                distributedData.identifier,
                distributedData.numRows,
                distributedData.numCols
            };
            MPIHelper::requestData(rank, info);
        }
        auto collectedDataItems = 0u;
        for (size_t i = 1; i < worldSize; i++) {
            size_t len;
            int rank;
            std::vector<char> buffer;
            MPIHelper::getMessage(&rank, TypesOfMessages::OUTPUT, MPI_UNSIGNED_CHAR, buffer, &len);
            
            std::string address = std::to_string(rank);  
            auto dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);   
                    
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }            

            auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buffer));
            auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
            auto slicedMatValues = slicedMat->getValues();
            for (size_t r = 0; r < dp->range->r_len; r++) {
                memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                resValues += denseMat->getRowSkip();
                slicedMatValues += slicedMat->getRowSkip();
            }
            DataObjectFactory::destroy(slicedMat);
            
            collectedDataItems+=  dp->range->r_len *  dp->range->c_len;

            auto distributedData = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();            
            distributedData.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(distributedData);
            // this is to handle the case when not all workers participate in the computation, i.e., number of workers is larger than of the work items
            if(collectedDataItems == denseMat->getNumRows() * denseMat->getNumCols())
                break;
        }
    };
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        

        struct StoredInfo{
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Data> caller(dctx);


        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
            StoredInfo storedInfo({dp->dp_id});
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);                       

            caller.asyncTransferCall(address, storedInfo, protoData);
        }
                
        

        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            auto dp = mat->getMetaDataObject()->getDataPlacementByID(dp_id);
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            

            auto matProto = response.result;
            
            // TODO: We need to handle different data types 
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            // Zero copy buffer
            std::vector<char> buf(static_cast<const char*>(matProto.bytes().data()), static_cast<const char*>(matProto.bytes().data()) + matProto.bytes().size()); 
            auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));
            auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
            auto slicedMatValues = slicedMat->getValues();
            for (size_t r = 0; r < dp->range->r_len; r++){
                memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                resValues += denseMat->getRowSkip();                    
                slicedMatValues += slicedMat->getRowSkip();
            }
            DataObjectFactory::destroy(slicedMat);

            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
        } 
    };
};


// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        

        auto ctx = DistributedContext::get(dctx);
        std::vector<std::thread> threads_vector;

        auto dpVector = mat->getMetaDataObject()->getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);                       

            std::thread t([address, dp = dp.get(), protoData, distributedData, &mat, &ctx]() mutable
            {
                auto stub = ctx->stubs[address].get();

                distributed::Data matProto;
                grpc::ClientContext grpc_ctx;
                stub->Transfer(&grpc_ctx, protoData, &matProto);
            
                // TODO: We need to handle different data types 
                auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
                if (!denseMat){
                    throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
                }
                // Zero copy buffer
                std::vector<char> buf(static_cast<const char*>(matProto.bytes().data()), static_cast<const char*>(matProto.bytes().data()) + matProto.bytes().size()); 
                auto slicedMat = dynamic_cast<DenseMatrix<double>*>(DF_deserialize(buf));
                auto resValues = denseMat->getValues() + (dp->range->r_start * denseMat->getRowSkip());
                auto slicedMatValues = slicedMat->getValues();
                for (size_t r = 0; r < dp->range->r_len; r++){
                    memcpy(resValues + dp->range->c_start, slicedMatValues, dp->range->c_len * sizeof(double));
                    resValues += denseMat->getRowSkip();                    
                    slicedMatValues += slicedMat->getRowSkip();
                }
                DataObjectFactory::destroy(slicedMat);
                
                distributedData.isPlacedAtWorker = false;
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(distributedData);
            });
            threads_vector.push_back(move(t));        
        }
        for (auto &thread : threads_vector)
            thread.join();
    };
};

