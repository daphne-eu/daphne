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

#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>

#include <cassert>
#include <cstddef>


template<class DT>
struct Broadcast
{
    static void apply(DT *&mat, bool isScalar, ALLOCATION_TYPE alloc_type, DCTX(ctx)) 
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
        double val;
        if (isScalar) {
            auto ptr = (double*)(&mat);
            val = *ptr;
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false);
        }
       
        Range range;
        range.r_start = 0;
        range.c_start = 0;
        range.r_len = mat->getNumRows();
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
                dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
            }
            else {  // else create new omd entry
                // Find alloc_type
                switch (alloc_type) {
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
        }       
        // Find alloc_type
        IAllocationDescriptorDistributed *backend;
        switch (alloc_type){
            case ALLOCATION_TYPE::DIST_GRPC:
                backend = new AllocationDescriptorDistributedGRPC();
                break;
            case ALLOCATION_TYPE::DIST_OPENMPI:
                throw std::runtime_error("MPI support missing");
                break;
            default:
                throw std::runtime_error("No distributed implementation found");
                break;
        }
        IAllocationDescriptorDistributed::DistributedResult results;
        if (isScalar)
            results = backend->Broadcast(&val, mat);
        else
            results = backend->Broadcast(mat);
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
    };           
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void broadcast(DT *&mat, bool isScalar, ALLOCATION_TYPE alloc_type, DCTX(ctx))
{
    Broadcast<DT>::apply(mat, isScalar, alloc_type, ctx);
}



#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
