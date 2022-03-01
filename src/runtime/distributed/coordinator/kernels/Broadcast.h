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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Broadcast
{
    static void apply(DT *mat, DCTX(ctx)) 
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

        assert(mat != nullptr);

        struct StoredInfo {
            std::string workerAddr;
        };
        DistributedCaller<StoredInfo, distributed::Matrix, distributed::StoredData> caller;
        

        distributed::Matrix protoMat;
        ProtoDataConverter<DT>::convertToProto(mat, &protoMat);
        
        for (auto i=0ul; i < workers.size(); i++){
            auto workerAddr = workers.at(i);

            StoredInfo storedInfo ({workerAddr});
            caller.asyncStoreCall(workerAddr, storedInfo, protoMat);
        
        }
        // get results
        DataPlacement::DistributedMap dataMap;
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();            
            auto workerAddr = response.storedInfo.workerAddr;            

            auto storedData = response.result;
            if (std::is_same<DT, DenseMatrix<double>>::value)
                storedData.set_type(distributed::StoredData::Type::StoredData_Type_DenseMatrix_f64);            
            if (std::is_same<DT, DenseMatrix<int64_t>>::value)
                storedData.set_type(distributed::StoredData::Type::StoredData_Type_DenseMatrix_i64);
            if (std::is_same<DT, CSRMatrix<double>>::value)
                storedData.set_type(distributed::StoredData::Type::StoredData_Type_CSRMatrix_f64);
            if (std::is_same<DT, CSRMatrix<int64_t>>::value)
                storedData.set_type(distributed::StoredData::Type::StoredData_Type_CSRMatrix_i64);
            
            DistributedData data(storedData);
            dataMap[workerAddr] = data;
        }
        DataPlacement dataPlacement(dataMap);
        dataPlacement.isPlacedOnWorkers = true;
        mat->dataPlacement = dataPlacement;        
    };           
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void broadcast(DT *mat, DCTX(ctx))
{
    Broadcast<DT>::apply(mat, ctx);
}



#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_BROADCAST_H
