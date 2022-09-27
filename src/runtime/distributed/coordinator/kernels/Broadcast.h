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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
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
    static void apply(DT *&mat, bool isScalar, DCTX(ctx))
    { 
        std::cout<<"MPI broadcast dense"<<std::endl;
        size_t messageLength;
        void * dataToSend= MPISerializer<DT>::serialize(mat, isScalar, &messageLength);
        MPIWorker::sendData(messageLength, dataToSend);
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
            // Need matrix for metadata, type of matrix does not really matter..
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
            auto protoVal = protoMsg.mutable_value();
            protoVal->set_f64(*val);
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

            DistributedData data;
            data.ix = DistributedIndex(0, 0);
            // If DataPlacement dp already exists simply
            // update range (in case we have a different one) and distributed data
            DataPlacement *dp;
            if ((dp = mat->getMetaDataObject().getDataPlacementByLocation(workerAddr))) {                
                mat->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
            }
            else {  // else create new dp entry
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


#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
