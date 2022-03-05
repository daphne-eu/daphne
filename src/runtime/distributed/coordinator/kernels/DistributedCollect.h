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

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorCombine;


// ****************************************************************************
// Template speciliazations for each communication framework (gRPC, OPENMPI, etc.)
// ****************************************************************************

template<DISTRIBUTED_BACKEND backend, class DT>
struct CollectImplementationClass
{
    static void apply(DT *&mat, DCTX(ctx));
};

template<DISTRIBUTED_BACKEND backend, class DT>
void collectImplementation(DT *&mat, DCTX(ctx)) 
{
    CollectImplementationClass<backend, DT>::apply(mat, ctx);
}

// ****************************************************************************
// gRPC implementation
// ****************************************************************************
template<class DT>
struct CollectImplementationClass<DISTRIBUTED_BACKEND::GRPC, DT>
{
    static void apply(DT *&mat, DCTX(ctx)) 
    {        
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
        DistributedCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;


        auto omdVector = mat->getObjectMetaDataByType(ALLOCATION_TYPE::DISTRIBUTED);
        for (auto &omd : *omdVector) {
            auto address = omd->allocation->getLocation();
            
            auto distributedData = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
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
            auto data = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();            

            auto matProto = response.result;
            ProtoDataConverter<DT>::convertFromProto(
                matProto, mat,
                omd->range->r_start, omd->range->r_start + omd->range->r_len,
                omd->range->c_start, omd->range->c_start + omd->range->c_len);                
            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
        }      
    }
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct DistributedCollect
{
    static void apply(DT *&mat, DCTX(ctx)) 
    {
       
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");
        // Is there anything to do that is framework agnostic?        
        collectImplementation<DISTRIBUTED_BACKEND::GRPC, DT>(mat, ctx);
  
    };
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCollect(DT *&mat, DCTX(ctx))
{
    DistributedCollect<DT>::apply(mat, ctx);
}

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H