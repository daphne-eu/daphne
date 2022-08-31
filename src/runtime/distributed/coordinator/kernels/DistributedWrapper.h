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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H

#include <ir/daphneir/Daphne.h>

#include <runtime/distributed/coordinator/kernels/Broadcast.h>
#include <runtime/distributed/coordinator/kernels/Distribute.h>
#include <runtime/distributed/coordinator/kernels/DistributedCollect.h>
#include <runtime/distributed/coordinator/kernels/DistributedCompute.h>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;
template <class DT>
class DistributedWrapper {};

template <typename VT>
class DistributedWrapper<DenseMatrix<VT>> {
private:
    DCTX(_ctx);

protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, const Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }
public:
    DistributedWrapper(DCTX(ctx)) : _ctx(ctx) {
        //TODO start workers from here instead of manually (e.g. resource manager) ? 

    }
    ~DistributedWrapper() = default; //TODO Terminate workers (e.g. with gRPC, resource manager, etc.)

    void execute(const char *mlirCode,
                 DenseMatrix<VT> ***res,
                 const Structure **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 size_t *outRows,
                 size_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines)                 
    {        
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        // assert(envVar && "Environment variable has to be set");
        // std::string workersStr(envVar);
        std::string workersStr("localhost:5000");    
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);


        // output allocation for row-wise combine
        for(size_t i = 0; i < numOutputs; ++i) {
            if(*(res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                *(res[i]) = DataObjectFactory::create<DenseMatrix<VT>>(outRows[i], outCols[i], zeroOut);
            }
        }
        
        // Distribute and broadcast inputs        
        // We create Handle_v2 object here and we pass it to each primitive.
        // Each primitive (i.e. distribute/broadcast) populates this handle with data information for each worker
        Handle_v2<Structure> *handle = new Handle_v2<Structure>(workers);    
        for (auto i = 0u; i < numInputs; ++i) {            
            if (isBroadcast(splits[i], inputs[i])){
                broadcast(handle, (const DenseMatrix<VT>*)inputs[i], _ctx);   
            }
            else {
                distribute(handle, (const DenseMatrix<VT>*)inputs[i], _ctx);
            }
        }
        Handle_v2<Structure> *resHandle = nullptr;
          
        //TODO This is hardcoded for now, will be deleted
        // std::string code (
        // "   func @dist(%arg0: !daphne.Matrix<?x?xf64>, %arg1: !daphne.Matrix<?x?xf64>, %arg2: !daphne.Matrix<?x?xf64>) -> (!daphne.Matrix<?x?xf64>) {\n "
        // "     %0 = \"daphne.numRows\"(%arg0) : (!daphne.Matrix<?x?xf64>) -> index\n "
        // "     %1 = \"daphne.numCols\"(%arg0) : (!daphne.Matrix<?x?xf64>) -> index\n "
        // "     %2 = \"daphne.ewMul\"(%arg0, %arg1) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n "
        // "     %3 = \"daphne.ewAdd\"(%2, %arg2) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n "
        // "     \"daphne.return\"(%3) : (!daphne.Matrix<?x?xf64>) -> ()\n "
        // "   }\n "
        // );
        // This seems to work better for now... TODO fix generated mlir code
        // "func @dist(%arg0: !daphne.Matrix<?x?xf64>, %arg1: !daphne.Matrix<?x?xf64>, %arg2: !daphne.Matrix<?x?xf64>, %arg3: !daphne.Matrix<?x?xf64>, %arg4: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {\n"
        // // // "     %0 = \"daphne.numRows\"(%arg0) : (!daphne.Matrix<?x?xf64>) -> index\n "
        // // // "     %1 = \"daphne.numCols\"(%arg0) : (!daphne.Matrix<?x?xf64>) -> index\n "
        //     "   %0 = \"daphne.ewMul\"(%arg0, %arg1) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
        //     "   %1 = \"daphne.ewAdd\"(%0, %arg2) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
        //     "   %2 = \"daphne.ewMul\"(%1, %arg3) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
        //     "   %3 = \"daphne.ewMul\"(%2, %arg4) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>\n"
        //     "\"daphne.return\"(%3) : (!daphne.Matrix<?x?xf64>) -> ()\n"
        //     "}"
        // );
        // mlirCode = code.c_str();
        
        // TODO numInputs is not needed anymore, we need to update distributedCompute primitive/kernel
        distributedCompute(resHandle, handle, numInputs, mlirCode, _ctx);

        // Collect
        // TODO check *combines for aggregations and use corresponding distributed primitives
        for (size_t o = 0; o < numOutputs; o++){
            if (combines[o] == VectorCombine::ROWS){
                distributedCollect(*res[0], o, resHandle, _ctx);
            }
            else {
                assert ("we only support rows collect at the moment");
            }
        }
        
        
    }
};

// TODO for CSR
template<typename VT>
class DistributedWrapper<CSRMatrix<VT>> {
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
