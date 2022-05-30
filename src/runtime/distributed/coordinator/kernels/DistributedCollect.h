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


#include <cassert>
#include <cstddef>
#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct DistributedCollect
{
    static void apply(DT *&mat, ALLOCATION_TYPE alloc_type, DCTX(ctx)) 
    {
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
        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");
        backend->Collect(mat);  
    };
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCollect(DT *&mat, ALLOCATION_TYPE alloc_type, DCTX(ctx))
{
    DistributedCollect<DT>::apply(mat, alloc_type, ctx);
}

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H