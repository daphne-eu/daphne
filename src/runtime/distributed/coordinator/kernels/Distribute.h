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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTE_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct Distribute {
    static void apply(DT *mat, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distribute(DT *mat, DCTX(ctx))
{
    Distribute<AT, DT>::apply(mat, ctx);
}


// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *mat, DCTX(ctx)) {
        struct StoredInfo {
            size_t dp_id;
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
    
        assert(mat != nullptr);

        auto r = 0ul;
        for (auto workerIx = 0ul; workerIx < workers.size() && r < mat->getNumRows(); workerIx++) {            
            auto workerAddr = workers.at(workerIx);                      

            auto k = mat->getNumRows() / workers.size();
            auto m = mat->getNumRows() % workers.size();            

            Range range;
            range.r_start = (workerIx * k) + std::min(workerIx, m);
            range.r_len = ((workerIx + 1) * k + std::min(workerIx + 1, m)) - range.r_start;
            range.c_start = 0;
            range.c_len = mat->getNumCols();
                        
            DistributedData data;
            data.ix = DistributedIndex(workerIx, 0);
            // If dp already exists simply
            // update range (in case we have a different one) and distribute data
            DataPlacement *dp;
            if ((dp = mat->mdo.getDataPlacementByLocation(workerAddr))) {                
                mat->mdo.updateRangeDataPlacementByID(dp->dp_id, &range);     
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
            }
            else { // Else, create new object metadata entry
                    AllocationDescriptorGRPC *allocationDescriptor;
                    allocationDescriptor = new AllocationDescriptorGRPC(
                                                ctx,
                                                workerAddr,
                                                data);
                dp = mat->mdo.addDataPlacement(allocationDescriptor, &range);                    
            }
            // keep track of proccessed rows
            // Skip if already placed at workers
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            distributed::Data protoMsg;
        

            // TODO: We need to handle different data types 
            // (this will be simplified when serialization is implemented)
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix(), 
                                                    range.r_start,
                                                    range.r_start + range.r_len,
                                                    range.c_start,
                                                    range.c_start + range.c_len);

            StoredInfo storedInfo({dp->dp_id}); 
            caller.asyncStoreCall(dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation(), storedInfo, protoMsg);
            r = (workerIx + 1) * k + std::min(workerIx + 1, m);
        }                
                       

        // get results       
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            
            auto storedData = response.result;            

            auto dp = mat->mdo.getDataPlacementByID(dp_id);
            
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
            data.identifier = storedData.identifier();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);            
        }

    }
};



#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTE_H
