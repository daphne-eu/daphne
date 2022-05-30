/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/worker/ProtoDataConverter.h>
#include <runtime/distributed/coordinator/kernels/DistributedGRPCCaller.h>

#include <ir/daphneir/Daphne.h>

IAllocationDescriptorDistributed::DistributedResult AllocationDescriptorDistributedGRPC::Distribute(const Structure *mat) 
{         
    struct StoredInfo {
        size_t omd_id;
    };                
    DistributedGRPCCaller<StoredInfo, distributed::Matrix, distributed::StoredData> caller;

    auto omdVector = (mat->getObjectMetaDataByType(ALLOCATION_TYPE::DIST_GRPC));
    for (auto &omd : *omdVector) {
        // Skip if already placed at workers
        if (dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData().isPlacedAtWorker)
            continue;
        distributed::Matrix protoMat;
        // TODO: We need to handle different data types 
        // (this will be simplified when serialization is implemented)
        auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
        if (!denseMat){
            std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
        }
        ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, &protoMat, 
                                                omd->range->r_start,
                                                omd->range->r_start + omd->range->r_len,
                                                omd->range->c_start,
                                                omd->range->c_start + omd->range->c_len);

        StoredInfo storedInfo({omd->omd_id}); 
        caller.asyncStoreCall(dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getLocation(), storedInfo, protoMat);
    }

    // get results       
    IAllocationDescriptorDistributed::DistributedResult results;
    std::map<size_t, IAllocationDescriptorDistributed::StoredInfo> map;
    while (!caller.isQueueEmpty()){
        auto response = caller.getNextResult();
        auto omd_id = response.storedInfo.omd_id;
        
        auto storedData = response.result;
        
        IAllocationDescriptorDistributed::StoredInfo info;
        info.filename = storedData.filename();
        info.numRows = storedData.num_rows();
        info.numCols = storedData.num_cols();
        map.insert({omd_id, info});
    }
    results.push_back(map);
    return results;
}

IAllocationDescriptorDistributed::DistributedResult AllocationDescriptorDistributedGRPC::Broadcast(const Structure *mat) 
{ 
    
    struct StoredInfo {
        size_t omd_id;
    };
    DistributedGRPCCaller<StoredInfo, distributed::Matrix, distributed::StoredData> caller;
    

    distributed::Matrix protoMat;
    auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
    if (!denseMat){
        std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
    }
    ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, &protoMat);
    auto omdVector = (mat->getObjectMetaDataByType(ALLOCATION_TYPE::DIST_GRPC));
    for (auto &omd : *omdVector) {
        if (dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData().isPlacedAtWorker)
            continue;
        auto addr = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getLocation();       
        StoredInfo storedInfo({omd->omd_id});
        caller.asyncStoreCall(addr, storedInfo, protoMat);
    }
    
    IAllocationDescriptorDistributed::DistributedResult results;
    // get results        
    std::map<size_t, IAllocationDescriptorDistributed::StoredInfo> map;
    while (!caller.isQueueEmpty()){

        IAllocationDescriptorDistributed::StoredInfo info;

        auto response = caller.getNextResult();            
        auto omd_id = response.storedInfo.omd_id;
        auto storedData = response.result;
        
        // storedData.set_type(dataType);
        info.filename = storedData.filename();
        info.numRows = storedData.num_rows();
        info.numCols = storedData.num_cols();
        
        map.insert({omd_id, info});
    }
    results.push_back(map);  
    return results;
}

IAllocationDescriptorDistributed::DistributedComputeResult AllocationDescriptorDistributedGRPC::Compute(const Structure **args, size_t numInputs, const char *mlirCode)
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
    struct StoredInfo {
        std::string addr;
    };                
    DistributedGRPCCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;

    // Broadcast at workers
    for (auto addr : workers) {                        
        distributed::Task task;
        
        // Pass all the nessecary arguments for the pipeline
        for (size_t i = 0; i < numInputs; i++) {
            // Find input for this worker
            auto omd = args[i]->getObjectMetaDataByLocation(addr);

            auto distrData = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
            distributed::StoredData protoData;
            protoData.set_filename(distrData.filename);
            protoData.set_num_rows(distrData.numRows);
            protoData.set_num_cols(distrData.numCols); 
            
            *task.add_inputs()->mutable_stored()= protoData;
        }
        task.set_mlir_code(mlirCode);           
        StoredInfo storedInfo({addr});    
        // TODO for now resuing channels seems to slow things down... 
        // It is faster if we generate channel for each call and let gRPC handle resources internally
        // We might need to change this in the future and re-use channels ( data.getChannel() )
        caller.asyncComputeCall(addr, storedInfo, task);
    }

    
    // Get Results
    IAllocationDescriptorDistributed::DistributedComputeResult results;    
    
    while (!caller.isQueueEmpty()){
        auto response = caller.getNextResult();
        auto addr = response.storedInfo.addr;
        
        auto computeResult = response.result;            
        
        for (int o = 0; o < computeResult.outputs_size(); o++){            

            IAllocationDescriptorDistributed::StoredInfo info;
            info.filename = computeResult.outputs()[o].stored().filename();
            info.numRows = computeResult.outputs()[o].stored().num_rows();
            info.numCols = computeResult.outputs()[o].stored().num_cols();
            
            if (results.size() <= (size_t)o)
                results.push_back(std::map<std::string, IAllocationDescriptorDistributed::StoredInfo>());
            results[o].insert({addr, info});
        }            
    }
    return results;
}

void AllocationDescriptorDistributedGRPC::Collect(Structure *mat) 
{ 
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

    struct StoredInfo{
        size_t omd_id;
    };
    DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Matrix> caller;


    auto omdVector = mat->getObjectMetaDataByType(ALLOCATION_TYPE::DIST_GRPC);
    for (auto &omd : *omdVector) {
        auto address = omd->allocation->getLocation();
        
        auto distributedData = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
        StoredInfo storedInfo({omd->omd_id});
        distributed::StoredData protoData;
        protoData.set_filename(distributedData.filename);
        protoData.set_num_rows(distributedData.numRows);
        protoData.set_num_cols(distributedData.numCols);                       

        caller.asyncTransferCall(address, storedInfo, protoData);
    }
            
    

    while (!caller.isQueueEmpty()){
        auto response = caller.getNextResult();
        auto omd_id = response.storedInfo.omd_id;
        auto omd = mat->getObjectMetaDataByID(omd_id);
        auto data = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();            

        auto matProto = response.result;
        
        auto denseMat = dynamic_cast<DenseMatrix<double>*>(mat);
        if (!denseMat){
            std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
        }        
        ProtoDataConverter<DenseMatrix<double>>::convertFromProto(
            matProto, denseMat,
            omd->range->r_start, omd->range->r_start + omd->range->r_len,
            omd->range->c_start, omd->range->c_start + omd->range->c_len);                
        data.isPlacedAtWorker = false;
        dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
    } 
}