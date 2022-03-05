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
#include <runtime/distributed/coordinator/datastructures/Handle.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTres, class DTarg>
struct DistributedCollect
{
    static void apply(DTres *&res, size_t resIdx, const Handle_v2<DTarg> *handle, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTres, class DTarg>
void distributedCollect(DTres *&res, size_t resIdx, const Handle_v2<DTarg> *handle, DCTX(ctx))
{
    DistributedCollect<DTres, DTarg>::apply(res, resIdx, handle, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct DistributedCollect<DenseMatrix<double>, Structure>
{
    static void apply(DenseMatrix<double> *&res, size_t resIdx, const Handle_v2<Structure> *handle, DCTX(ctx))
    {
        struct StoredInfo{
            DistributedIndex *ix;
            std::string workerAddr;
            std::shared_ptr<grpc::Channel> channel;
        };
        DistributedCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;

        assert (res != nullptr && "result matrix must be already allocated by wrapper since only there exists information regarding size");

        for (auto &pair : handle->getMap()) {
            auto ix = pair.first;
            // Collect specified result index
            auto data = pair.second[resIdx];

            // TODO this is uneccesary
            StoredInfo storedInfo ({new DistributedIndex(0, 0)});
            caller.asyncTransferCall(data.getAddress(), storedInfo, data.getData());
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
        auto k = res->getNumRows() / workersSize;
        auto m = res->getNumRows() % workersSize;        
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto ix = response.storedInfo.ix;
            auto matProto = response.result;
            ProtoDataConverter::convertFromProto(matProto,
                res,
                ix->getRow() * k + std::min(ix->getRow(), m),
                (ix->getRow() + 1) * k + std::min((ix->getRow() + 1), m),
                0,
                res->getNumCols());
        }      
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOLLECT_H