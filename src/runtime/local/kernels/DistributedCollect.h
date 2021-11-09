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
struct DistributedCollect
{
    static void apply(DT *&res, const Handle<DT> *handle, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCollect(DT *&res, const Handle<DT> *handle, DCTX(ctx))
{
    DistributedCollect<DT>::apply(res, handle, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct DistributedCollect<DenseMatrix<double>>
{
    static void apply(DenseMatrix<double> *&res, const Handle<DenseMatrix<double>> *handle, DCTX(ctx))
    {
        // ****************************************************************************
        // Struct for async calls
        // ****************************************************************************

        struct AsyncClientCall {         
            const DistributedIndex *ix;  
            distributed::Matrix matProto;

            grpc::ClientContext context;

            grpc::Status status;

            std::unique_ptr<grpc::ClientAsyncResponseReader<distributed::Matrix>> response_reader;
        };
        grpc::CompletionQueue cq;
        auto callCounter = 0;
        auto blockSize = DistributedData::BLOCK_SIZE;
        res = DataObjectFactory::create<DenseMatrix<double>>(handle->getRows(), handle->getCols(), false);
        for (auto &pair : handle->getMap()) {
            auto ix = pair.first;
            auto data = pair.second;

            auto stub = distributed::Worker::NewStub(data.getChannel());
            grpc::ClientContext context;

            distributed::Matrix matProto;
            
            
            struct AsyncClientCall *call = new AsyncClientCall;
            call->ix = new DistributedIndex(ix);
            call->response_reader = stub->AsyncTransfer(&call->context, data.getData(), &cq);
            call->response_reader->Finish(&call->matProto, &call->status, (void*)call);
            callCounter++;
        }
        void *got_tag;
        bool ok = false;
        distributed::ComputeResult result;
        while(cq.Next(&got_tag, &ok)){
            callCounter--;
            AsyncClientCall *call = static_cast<AsyncClientCall*>(got_tag);
            if (!ok) {
                throw std::runtime_error(
                    call->status.error_message()
                );
            }
            ProtoDataConverter::convertFromProto(call->matProto,
                res,
                call->ix->getRow() * blockSize,
                std::min((call->ix->getRow() + 1) * blockSize, res->getNumRows()),
                call->ix->getCol() * blockSize,
                std::min((call->ix->getCol() + 1) * blockSize, res->getNumCols()));
            // This is needed. Cq.Next() never returns if there are no elements to read from completition queue cq.
            if (callCounter == 0)
                break;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOLLECT_H