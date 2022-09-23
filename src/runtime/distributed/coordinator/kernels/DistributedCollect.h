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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

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
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *&mat, DCTX(dctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");        

        struct StoredInfo{
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;


        auto dpVector = mat->getMetaDataObject().getDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
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
            auto dp = mat->getMetaDataObject().getDataPlacementByID(dp_id);
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();            

            auto matProto = response.result;
            
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }        
            ProtoDataConverter<DenseMatrix<double>>::convertFromProto(
                matProto, denseMat,
                dp->range->r_start, dp->range->r_start + dp->range->r_len,
                dp->range->c_start, dp->range->c_start + dp->range->c_len);                
            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
        } 
    };
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H