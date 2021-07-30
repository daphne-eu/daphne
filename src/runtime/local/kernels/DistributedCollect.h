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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOLLECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOLLECT_H

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
struct DistributedCollect
{
    static void apply(DT *&res, const Handle<DT> *handle) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCollect(DT *&res, const Handle<DT> *handle)
{
    DistributedCollect<DT>::apply(res, handle);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct DistributedCollect<DenseMatrix<double>>
{
    static void apply(DenseMatrix<double> *&res, const Handle<DenseMatrix<double>> *handle)
    {
        auto blockSize = DistributedData::BLOCK_SIZE;
        res = DataObjectFactory::create<DenseMatrix<double>>(handle->getRows(), handle->getCols(), false);
        for (auto &pair : handle->getMap()) {
            auto ix = pair.first;
            auto data = pair.second;

            auto stub = distributed::Worker::NewStub(data.getChannel());
            grpc::ClientContext context;

            distributed::Matrix matProto;
            auto status = stub->Transfer(&context, data.getData(), &matProto);
            if (!status.ok()) {
                throw std::runtime_error(
                    status.error_message()
                );
            }

            ProtoDataConverter::convertFromProto(matProto,
                res,
                ix.getRow() * blockSize,
                std::min((ix.getRow() + 1) * blockSize, res->getNumRows()),
                ix.getCol() * blockSize,
                std::min((ix.getCol() + 1) * blockSize, res->getNumCols()));
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOLLECT_H