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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOMPUTE_H
#define SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOMPUTE_H

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
struct DistributedCompute
{
    static void apply(Handle<DT> *&res, const Handle<DT> **args, size_t num_args, const char *mlirCode, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCompute(Handle<DT> *&res, const Handle<DT> **args, size_t num_args, const char *mlirCode, DCTX(ctx))
{
    DistributedCompute<DT>::apply(res, args, num_args, mlirCode, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct DistributedCompute<DenseMatrix<double>>
{
    static void apply(Handle<DenseMatrix<double>> *&res,
                      const Handle<DenseMatrix<double>> **args,
                      size_t num_args,
                      const char *mlirCode,
                      DCTX(ctx))
    {

        // ****************************************************************************
        // Struct for async calls
        // ****************************************************************************

        struct AsyncClientCall {
            const DistributedIndex *ix;
            DistributedData *data;

            distributed::ComputeResult result;

            grpc::ClientContext context;

            grpc::Status status;

            std::unique_ptr<grpc::ClientAsyncResponseReader<distributed::ComputeResult>> response_reader;
        };
        grpc::CompletionQueue cq;
        auto callCounter = 0;
        assert(num_args == 2 && "Only binary supported for now");
        auto lhs = args[0];
        auto rhs = args[1];

        Handle<DenseMatrix<double>>::HandleMap resMap;
        for (auto &pair : lhs->getMap()) {
            auto ix = pair.first;
            auto lhsData = pair.second;
            auto rhsData = rhs->getMap().at(ix);

            if (lhsData.getAddress() == rhsData.getAddress()) {
                // data is on same worker -> direct execution possible
                auto stub = distributed::Worker::NewStub(lhsData.getChannel());

                grpc::ClientContext context;

                distributed::Task task;
                *task.add_inputs()->mutable_stored() = lhsData.getData();
                *task.add_inputs()->mutable_stored() = rhsData.getData();
                task.set_mlir_code(mlirCode);
                
                struct AsyncClientCall *call = new AsyncClientCall;
                call->data = new DistributedData(lhsData);
                call->ix = new DistributedIndex(ix);

                call->response_reader = stub->AsyncCompute(&call->context, task, &cq);
                call->response_reader->Finish(&call->result, &call->status, (void*)call);

                callCounter++;
            }
            else {
                // TODO: send data between workers
                throw std::runtime_error(
                    "Data shuffling not yet supported"
                );
            }
        }
        void *got_tag;
        bool ok = false;
        while(cq.Next(&got_tag, &ok)){    
            callCounter--;
            AsyncClientCall *call = static_cast<AsyncClientCall*>(got_tag);
            if (!ok) {
                throw std::runtime_error(
                    call->status.error_message()
                );
            }
            DistributedData data(call->result.outputs(0).stored(), call->data->getAddress(), call->data->getChannel());        
            resMap.insert({*call->ix, data});
            // This is needed. Cq.Next() never returns if there are no elements to read from completition queue cq.
            if (callCounter == 0)
                break;
        }        
        res = new Handle<DenseMatrix<double>>(resMap, lhs->getRows(), lhs->getCols());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOMPUTE_H