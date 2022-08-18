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

#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>
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
    static void apply(DT *&mat, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distributedCollect(DT *&mat, DCTX(ctx))
{
    DistributedCollect<AT, DT>::apply(mat, ctx);
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
    static void apply(DT *&mat, DCTX(ctx)) 
    {
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");
        // Get num workers
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);
        std::string delimiter(",");

        size_t pos;
        auto workersSize = 0;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workersSize++;
            workersStr.erase(0, pos + delimiter.size());
        }
        workersSize++;

        struct StoredInfo{
            size_t omd_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;


        auto omdVector = mat->getObjectMetaDataByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &omd : *omdVector) {
            auto address = omd->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).getDistributedData();
            StoredInfo storedInfo({omd->omd_id});
            distributed::StoredData protoData;
            protoData.set_filename(distributedData.filename);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);                       

            caller.asyncTransferCall(address, storedInfo, protoData);
        }
                
        

        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto omd_id = response.storedInfo.omd_id;
            auto omd = mat->getObjectMetaDataByID(omd_id);
            auto data = dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).getDistributedData();            

            auto matProto = response.result;
            
            auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }        
            ProtoDataConverter<DenseMatrix<double>>::convertFromProto(
                matProto, denseMat,
                omd->range->r_start, omd->range->r_start + omd->range->r_len,
                omd->range->c_start, omd->range->c_start + omd->range->c_len);                
            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).updateDistributedData(data);
        } 
    };
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H