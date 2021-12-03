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
        assert((num_args == 1 || num_args == 2)&& "Only binary and unary supported for now");
        struct StoredInfo {
            DistributedIndex *ix;
            DistributedData *data;
        };
        DistributedCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;
        Handle<DenseMatrix<double>>::HandleMap resMap;        
        size_t resultRows, resultColumns;
        // ****************************************************************************
        // Unary operations
        // ****************************************************************************
        // Tailored to row-wise aggregation.
        if (num_args == 1){
            auto arg = args[0];
            resultRows = arg->getRows();
            resultColumns = 1;
            for (auto &pair : arg->getMap()){
                auto ix = pair.first;
                auto argData = pair.second;
                distributed::Task task;
                *task.add_inputs()->mutable_stored() = argData.getData();
                task.set_mlir_code(mlirCode);
                
                StoredInfo storedInfo ({new DistributedIndex(ix), new DistributedData(argData)});

                caller.addAsyncCall(argData.getChannel(), storedInfo, task);
            }
        }
        // ****************************************************************************
        // Binary operations
        // ****************************************************************************
        // Tailored to element-wise binary operations.
        if (num_args == 2){
            auto lhs = args[0];
            auto rhs = args[1];

            const size_t numRowsLhs = lhs->getRows();
            const size_t numColsLhs = lhs->getCols();
            const size_t numRowsRhs = rhs->getRows();
            const size_t numColsRhs = rhs->getCols();

            // ****************************************************************************
            // Handle <- Matrix, Matrix
            // ****************************************************************************
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                resultRows = numRowsLhs;
                resultColumns = numColsLhs;
                for (auto &pair : lhs->getMap()) {
                    auto ix = pair.first;
                    auto lhsData = pair.second;
                    auto rhsData = rhs->getMap().find(ix)->second;

                    if (lhsData.getAddress() == rhsData.getAddress()) {                

                        distributed::Task task;
                        *task.add_inputs()->mutable_stored() = lhsData.getData();
                        *task.add_inputs()->mutable_stored() = rhsData.getData();
                        task.set_mlir_code(mlirCode);
                        
                        StoredInfo storedInfo ({new DistributedIndex(ix), new DistributedData(lhsData)});

                        caller.addAsyncCall(lhsData.getChannel(), storedInfo, task);
                    }
                    else {
                        // TODO: send data between workers
                        throw std::runtime_error(
                            "Data shuffling not yet supported"
                        );
                    }
                }
            }
            // ****************************************************************************
            // Handle <- Matrix, row
            // ****************************************************************************
            else if(numRowsRhs == 1 && numColsLhs == numColsRhs) {
                resultRows = numRowsLhs;
                resultColumns = numColsLhs;
                for (auto &pair : lhs->getMap()) {
                    auto ix = pair.first;
                    auto lhsData = pair.second;
                    DistributedData *rhsDataPtr = nullptr;
                    for (auto &pairRhs : rhs->getMap()){
                        if (pairRhs.second.getAddress() == lhsData.getAddress()) {
                            rhsDataPtr = new DistributedData(pairRhs.second);
                            break;
                        }
                    }
                    auto rhsData = DistributedData(*rhsDataPtr);
                    if (lhsData.getAddress() == rhsData.getAddress()) {                

                        distributed::Task task;
                        *task.add_inputs()->mutable_stored() = lhsData.getData();
                        *task.add_inputs()->mutable_stored() = rhsData.getData();
                        task.set_mlir_code(mlirCode);
                    
                        StoredInfo storedInfo ({new DistributedIndex(ix), new DistributedData(lhsData)});

                        caller.addAsyncCall(lhsData.getChannel(), storedInfo, task);
                    }
                    else {
                        // TODO: send data between workers
                        throw std::runtime_error(
                            "Data shuffling not yet supported"
                        );
                    }
                }
            }
            else {
                assert(
                        false && "lhs and rhs must either have the same dimensions, "
                        "or rhs be a row/column vector with the "
                        "width/height of the other"
                );
            }
        }
        // Get Results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto ix = response.storedInfo.ix;
            auto lhsdata = response.storedInfo.data;
            
            auto computeResult = response.result;
            DistributedData data(computeResult.outputs(0).stored(), lhsdata->getAddress(), lhsdata->getChannel());
            resMap.insert({*ix, data});
        }
        res = new Handle<DenseMatrix<double>>(resMap, resultRows, resultColumns);        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_DISTRIBUTEDCOMPUTE_H