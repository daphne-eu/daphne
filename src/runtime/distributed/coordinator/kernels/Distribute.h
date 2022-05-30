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

#include <cassert>
#include <cstddef>
#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Distribute
{
    static void apply(DT *mat, ALLOCATION_TYPE alloc_type, DCTX(ctx)) {
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
                        
            IAllocationDescriptor *allocationDescriptor;
            DistributedData data;
            data.ix = DistributedIndex(workerIx, 0);
            // If omd already exists simply
            // update range (in case we have a different one) and distributed data
            if (auto omd = mat->getObjectMetaDataByLocation(workerAddr)) {
                // TODO consider declaring objectmetadata functions const and objectmetadata array as mutable
                const_cast<typename std::remove_const<DT>::type*>(mat)->updateRangeObjectMetaDataByID(omd->omd_id, &range);     
                dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
            }
            else { // Else, create new object metadata entry
                // Find alloc_type
                switch (alloc_type){
                    case ALLOCATION_TYPE::DIST_GRPC:
                        allocationDescriptor = new AllocationDescriptorDistributedGRPC(
                                                    ctx, 
                                                    workerAddr,
                                                    data);            
                        break;
                    case ALLOCATION_TYPE::DIST_OPENMPI:
                        std::runtime_error("MPI support missing");
                        break;
                    default:
                        std::runtime_error("No distributed implementation found");
                        break;    
                }
                const_cast<typename std::remove_const<DT>::type*>(mat)->addObjectMetaData(allocationDescriptor, &range);                    
            }
            // keep track of proccessed rows
            r = (workerIx + 1) * k + std::min(workerIx + 1, m);
        }                
        // Find alloc_type
        IAllocationDescriptorDistributed *backend;
        switch (alloc_type){
            case ALLOCATION_TYPE::DIST_GRPC:
                backend = new AllocationDescriptorDistributedGRPC();
                break;
            case ALLOCATION_TYPE::DIST_OPENMPI:
                std::runtime_error("MPI support missing");
                break;
                    
            default:
                std::runtime_error("No distributed implementation found");
                break;
        }
        auto results = backend->Distribute(mat);
        for (auto &output : results){ 
            for (auto &workerResponse : output) {
                auto omd_id = workerResponse.first;
                auto storedInfo = workerResponse.second;
                auto omd = mat->getObjectMetaDataByID(omd_id);
                
                auto data = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
                data.filename = storedInfo.filename;
                data.numRows = storedInfo.numRows;
                data.numCols = storedInfo.numCols;
                data.isPlacedAtWorker = true;
                dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
            }
        }

    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distribute(DT *mat, ALLOCATION_TYPE alloc_type, DCTX(ctx))
{
    Distribute<DT>::apply(mat, alloc_type, ctx);
}


#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTE_H
