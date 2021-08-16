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

        auto workerIx = 0ul;
        auto blockSize = DistributedData::BLOCK_SIZE;

        Handle<DenseMatrix<double>>::HandleMap map;
        for (auto r = 0ul; r < (mat->getNumRows() - 1) / blockSize + 1; ++r) {
            for (auto c = 0ul; c < (mat->getNumCols() - 1) / blockSize + 1; ++c) {
                auto workerAddr = workers.at(workerIx);

                // TODO: reuse channels for workers
                auto channel = grpc::CreateChannel(workerAddr, grpc::InsecureChannelCredentials());
                auto stub = distributed::Worker::NewStub(channel);

                grpc::ClientContext context;

                distributed::Matrix protoMat;
                ProtoDataConverter::convertToProto(mat,
                    &protoMat,
                    r * blockSize,
                    std::min((r + 1) * blockSize, mat->getNumRows()),
                    c * blockSize,
                    std::min((c + 1) * blockSize, mat->getNumCols()));

                distributed::StoredData storedData;
                auto status = stub->Store(&context, protoMat, &storedData);

                if (!status.ok()) {
                    throw std::runtime_error(
                        status.error_message()
                    );
                }

                DistributedIndex ix(r, c);
                DistributedData data(storedData, workerAddr, channel);
                map.insert({ix, data});

                if (++workerIx == workers.size())
                    workerIx = 0;
            }
        }
        res = new Handle<DenseMatrix<double>>(map, mat->getNumRows(), mat->getNumCols());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTE_H