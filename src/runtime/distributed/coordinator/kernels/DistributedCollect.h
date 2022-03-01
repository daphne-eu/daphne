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
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct DistributedCollect
{
    static void apply(DT *&mat, DCTX(ctx)) 
    {
        struct StoredInfo{
            DistributedIndex *ix;
        };
        DistributedCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;

        assert (mat != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");

        auto dataPlacement = mat->dataPlacement;
        assert (dataPlacement.isPlacedOnWorkers && "in order to collect matrix must be placed on workers");

        auto dataMap = dataPlacement.getMap();
        for (auto &pair : dataMap) {
            auto address = pair.first;
            // Collect specified result index
            auto data = pair.second;

            StoredInfo storedInfo;
            storedInfo.ix = new DistributedIndex(data.getDistributedIndex());

            caller.asyncTransferCall(address, storedInfo, data.getData());
        }
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
        // Get Results
        auto combineType = mat->dataPlacement.combineType;
        size_t k = 0, m = 0;
        if (combineType == VectorCombine::ROWS) {
            k = mat->getNumRows() / workersSize;
            m = mat->getNumRows() % workersSize;
        }
        else if (combineType == VectorCombine::COLS){
            k = mat->getNumCols() / workersSize;
            m = mat->getNumCols() % workersSize;
        }
        else
            assert(!"Only Rows/Cols combineType supported atm");
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto ix = response.storedInfo.ix;
            auto matProto = response.result;
            if (combineType == VectorCombine::ROWS) {
                ProtoDataConverter<DT>::convertFromProto(matProto,
                    mat,
                    ix->getRow() * k + std::min(ix->getRow(), m),
                    (ix->getRow() + 1) * k + std::min((ix->getRow() + 1), m),
                    0,
                    mat->getNumCols());
            }
            else if (combineType == VectorCombine::COLS) {
                ProtoDataConverter<DT>::convertFromProto(matProto,
                    mat,
                    0,
                    mat->getNumRows(),
                    ix->getCol() * k + std::min(ix->getCol(), m),
                    (ix->getCol() + 1) * k + std::min((ix->getCol() + 1), m));
            }
            mat->dataPlacement.isPlacedOnWorkers = false;
        }      
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