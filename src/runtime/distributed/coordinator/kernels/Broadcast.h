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

#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct Broadcast {
    static void apply(DT *mat, bool isScalar, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void broadcast(DT *&mat, bool isScalar, DCTX(ctx))
{
    Broadcast<AT, DT>::apply(mat, isScalar, ctx);
}


// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *&mat, bool isScalar, DCTX(ctx)) 
    {
        struct StoredInfo {
            size_t omd_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller;
        
        
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

        distributed::Data protoMsg;

        assert(mat != nullptr && "Matrix to broadcast is nullptr");
        double *val;
        if (isScalar) {
            auto ptr = (double*)(&mat);
            val = ptr;
            // Need matrix for metadata
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
            // If omd already exists simply
            // update range (in case we have a different one) and distributed data
            ObjectMetaData *omd;
            if ((omd = mat->getObjectMetaDataByLocation(workerAddr))) {
                // TODO consider declaring objectmetadata functions const and objectmetadata array as mutable
                const_cast<typename std::remove_const<DT>::type*>(mat)->updateRangeObjectMetaDataByID(omd->omd_id, &range);
                dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).updateDistributedData(data);
            }
            else {  // else create new omd entry
                AllocationDescriptorDistributedGRPC *allocationDescriptor;
                allocationDescriptor = new AllocationDescriptorDistributedGRPC(
                                                ctx, 
                                                workerAddr,  
                                                data);
                omd = const_cast<typename std::remove_const<DT>::type*>(mat)->addObjectMetaData(allocationDescriptor, &range);
            }
            if (dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            
            StoredInfo storedInfo({omd->omd_id});
            caller.asyncStoreCall(workerAddr, storedInfo, protoMsg);
        }       
        
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();            
            auto omd_id = response.storedInfo.omd_id;
            auto omd = mat->getObjectMetaDataByID(omd_id);

            auto data = dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).getDistributedData();

            auto storedData = response.result;
            data.filename = storedData.filename();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;

            dynamic_cast<AllocationDescriptorDistributedGRPC&>(*(omd->allocation)).updateDistributedData(data);            
        }                
    };           
};


#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
