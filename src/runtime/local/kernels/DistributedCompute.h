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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Handle.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cblas.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct DistributedCompute
{
    static void apply(Handle<DT> *&res, const Handle<DT> *lhs, const Handle<DT> *rhs) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void distributedCompute(Handle<DT> *&res, const Handle<DT> *lhs, const Handle<DT> *rhs)
{
    DistributedCompute<DT>::apply(res, lhs, rhs);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<>
struct DistributedCompute<DenseMatrix<double>>
{
    static void apply(Handle<DenseMatrix<double>> *&res,
                      const Handle<DenseMatrix<double>> *lhs,
                      const Handle<DenseMatrix<double>> *rhs)
    {
        assert(lhs->getMap().size() == rhs->getMap().size() && "Number of keys/data have to match");

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
                // FIXME: in `RewriteToCallKernelOpPass` create string for body of `DistributedComputeOp` and pass it as
                //  an additional parameter to this kernel -> string representation necessary!
                task.set_mlir_code(
                    "func @dist(%arg0: !daphne.Matrix<f64>, %arg1: !daphne.Matrix<f64>) -> !daphne.Matrix<f64> {\n"
                    "      %13 = \"daphne.ewAdd\"(%arg0, %arg1) : (!daphne.Matrix<f64>, !daphne.Matrix<f64>) -> !daphne.Matrix<f64>\n"
                    "      \"daphne.return\"(%13) : (!daphne.Matrix<f64>) -> ()\n"
                    "    }");
                distributed::ComputeResult result;
                auto status = stub->Compute(&context, task, &result);

                if (!status.ok()) {
                    throw std::runtime_error(
                        status.error_message()
                    );
                }

                assert(result.outputs_size() == 1);

                DistributedData data(result.outputs(0).stored(), lhsData.getAddress(), lhsData.getChannel());
                resMap.insert({ix, data});
            }
            else {
                // TODO: send data between workers
                throw std::runtime_error(
                    "Data shuffling not yet supported"
                );
            }
        }
        res = new Handle<DenseMatrix<double>>(resMap, lhs->getRows(), lhs->getCols());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOMPUTE_H