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

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/local/datastructures/AllocationDescriptorDistributed.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>


// ****************************************************************************
// Template speciliazations for each communication framework (gRPC, OPENMPI, etc.)
// ****************************************************************************

template<DISTRIBUTED_BACKEND backend, class DT>
struct BroadcastImplementationClass
{
    static void apply(DT* mat) = delete;
};

template<DISTRIBUTED_BACKEND backend, class DT>
void broadcastImplementation(DT* mat)
{
    BroadcastImplementationClass<backend, DT>::apply(mat);
}
// ****************************************************************************
// gRPC implementation
// ****************************************************************************
template<class DT>
struct BroadcastImplementationClass<DISTRIBUTED_BACKEND::GRPC, DT>
{
    static void apply(DT *mat)
    {   
        struct StoredInfo {
            size_t omd_id;
        };
        DistributedCaller<StoredInfo, distributed::Matrix, distributed::StoredData> caller;
        

        distributed::Matrix protoMat;
        auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
        if (!denseMat){
            std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
        }
        ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, &protoMat);
        auto omdVector = (mat->getObjectMetaDataByType(ALLOCATION_TYPE::DISTRIBUTED));
        for (auto &omd : *omdVector) {
            if (dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            auto addr = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getLocation();       
            StoredInfo storedInfo({omd->omd_id});
            caller.asyncStoreCall(addr, storedInfo, protoMat);
        }
        
        // get results        
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();            
            auto omd_id = response.storedInfo.omd_id;
            auto omd = mat->getObjectMetaDataByID(omd_id);
            auto storedData = response.result;
            auto ix = dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedIndex();  
            
            // storedData.set_type(dataType);
            DistributedData data;
            data.ix = ix;
            data.filename = storedData.filename();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
        }    
    }
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Broadcast
{
    static void apply(DT *mat, DCTX(ctx)) 
    {
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);

        assert(mat != nullptr);

        Range range;
        range.r_start = 0;
        range.r_len = mat->getNumRows();
        range.c_start = 0;
        range.c_len = mat->getNumCols();
        for (auto i=0ul; i < workers.size(); i++){
            auto workerAddr = workers.at(i);

            IAllocationDescriptor *allocationDescriptor;
            DistributedData data;
            data.ix = DistributedIndex(0, 0);
            // If omd already exists simply
            // update range (in case we have a different one) and distributed data
            if (auto omd = mat->getObjectMetaDataByLocation(workerAddr)) {
                // TODO consider declaring objectmetadata functions const and objectmetadata array as mutable
                const_cast<typename std::remove_const<DT>::type*>(mat)->updateRangeObjectMetaDataByID(omd->omd_id, &range);
                dynamic_cast<AllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
            }
            else {  // else create new omd entry
                allocationDescriptor = new AllocationDescriptorDistributed(
                                                ctx, 
                                                workerAddr,  
                                                data); 
                const_cast<typename std::remove_const<DT>::type*>(mat)->addObjectMetaData(allocationDescriptor, &range);
            }
        }       
        broadcastImplementation<DISTRIBUTED_BACKEND::GRPC, DT>(mat);
    };           
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void broadcast(DT *mat, DCTX(ctx))
{
    Broadcast<DT>::apply(mat, ctx);
}



#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
