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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTE_H
#define SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Handle.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>
#include <runtime/local/kernels/DistributedCaller.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Distribute
{
    static void apply(Handle<DT> *&res, const DT *mat, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distribute(Handle<DT> *&res, const DT *mat, DCTX(ctx))
{
    Distribute<DT>::apply(res, mat, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct Distribute<DenseMatrix<double>>
{
    static void apply(Handle<DenseMatrix<double>> *&res, const DenseMatrix<double> *mat, DCTX(ctx))
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

        // auto blockSize = DistributedData::BLOCK_SIZE;

        struct StoredInfo {
            DistributedIndex *ix ;
            std::string workerAddr;
            std::shared_ptr<grpc::Channel> channel;
        };        
        DistributedCaller<StoredInfo, distributed::Matrix, distributed::StoredData> caller;

        Handle<DenseMatrix<double>>::HandleMap map;

        auto r = 0ul;
        for (auto workerIx = 0ul; workerIx < workers.size() && r < mat->getNumRows(); workerIx++) {            
            auto workerAddr = workers.at(workerIx);          

            distributed::Matrix protoMat;

            auto k = mat->getNumRows() / workers.size();
            auto m = mat->getNumRows() % workers.size();
            ProtoDataConverter::convertToProto(mat,
                &protoMat,
                (workerIx * k) + std::min(workerIx, m),
                (workerIx + 1) * k + std::min(workerIx + 1, m),
                0,
                mat->getNumCols());
                        
            auto channel = caller.GetOrCreateChannel(workerAddr);
            StoredInfo storedInfo ({new DistributedIndex(workerIx, 0), workerAddr, channel});
            caller.addAsyncCall(workerAddr, storedInfo, protoMat);
            
            // keep track of proccessed rows
            r = (workerIx + 1) * k + std::min(workerIx + 1, m);
        }
        // get results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto ix = response.storedInfo.ix;
            auto workerAddr = response.storedInfo.workerAddr;
            auto channel = response.storedInfo.channel;

            auto storedData = response.result;

            DistributedData data(storedData, workerAddr, channel);
            map.insert({*ix, data});
        }
        res = new Handle<DenseMatrix<double>>(map, mat->getNumRows(), mat->getNumCols());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTE_H