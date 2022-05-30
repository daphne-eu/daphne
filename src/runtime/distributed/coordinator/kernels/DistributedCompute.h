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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <cassert>
#include <cstddef>

using mlir::daphne::VectorCombine;



// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArgs>
struct DistributedCompute
{
    static void apply(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, ALLOCATION_TYPE alloc_type, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArgs>
void distributedCompute(DTRes **&res, size_t numOutputs, DTArgs **args, size_t numInputs, const char *mlirCode, VectorCombine *vectorCombine, ALLOCATION_TYPE alloc_type, DCTX(ctx))
{
    DistributedCompute<DTRes, DTArgs>::apply(res, numOutputs, args, numInputs, mlirCode, vectorCombine, alloc_type, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<class DTRes>
struct DistributedCompute<DTRes, const Structure>
{
    static void apply(DTRes **&res,
                      size_t numOutputs,
                      const Structure **args,
                      size_t numInputs,
                      const char *mlirCode,
                      VectorCombine *vectorCombine,
                      ALLOCATION_TYPE alloc_type,
                      DCTX(ctx))
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
        

        // Initialize Distributed index array, needed for results
        DistributedIndex *ix[numOutputs];
        for (size_t i = 0; i <numOutputs; i++)
            ix[i] = new DistributedIndex(0, 0);
        

        
        
         // Iterate over workers
        // Pass all the nessecary arguments for the pipeline
        for (auto addr : workers) {
            for (size_t i = 0; i < numOutputs; i++){                 
                // Get Result ranges
                auto combineType = vectorCombine[i];
                auto workersSize = workers.size();
                size_t k = 0, m = 0;                
                if (combineType == VectorCombine::ROWS) {
                    k = (*res[i])->getNumRows() / workersSize;
                    m = (*res[i])->getNumRows() % workersSize;
                }
                else if (combineType == VectorCombine::COLS){
                    k = (*res[i])->getNumCols() / workersSize;
                    m = (*res[i])->getNumCols() % workersSize;
                }
                else
                    assert(!"Only Rows/Cols combineType supported atm");

                IAllocationDescriptor *allocationDescriptor;
                DistributedData data;
                data.ix = *ix[i];
                data.vectorCombine = vectorCombine[i];
                data.isPlacedAtWorker = true;
                
                // Find alloc_type
                switch (alloc_type){
                    case ALLOCATION_TYPE::DIST_GRPC:
                        allocationDescriptor = new AllocationDescriptorDistributedGRPC(
                                                ctx,
                                                addr,
                                                data);
                        break;
                    case ALLOCATION_TYPE::DIST_OPENMPI:
                        std::runtime_error("MPI support missing");
                        break;
                            
                    default:
                        std::runtime_error("No distributed implementation found");
                        break;
                }  
                
                // Update distributed index for next iteration
                // and set ranges for objmetadata
                Range range;
                if (vectorCombine[i] == VectorCombine::ROWS) {
                    ix[i] = new DistributedIndex(ix[i]->getRow() + 1, ix[i]->getCol());            
                    
                    range.r_start = data.ix.getRow() * k + std::min(data.ix.getRow(), m);
                    range.r_len = ((data.ix.getRow() + 1) * k + std::min((data.ix.getRow() + 1), m)) - range.r_start;
                    range.c_start = 0;
                    range.c_len = (*res[i])->getNumCols();
                }
                if (vectorCombine[i] == VectorCombine::COLS) {
                    ix[i] = new DistributedIndex(ix[i]->getRow(), ix[i]->getCol() + 1);
                    
                    range.r_start = 0; 
                    range.r_len = (*res[i])->getNumRows(); 
                    range.c_start = data.ix.getCol() * k + std::min(data.ix.getCol(), m);
                    range.c_len = ((data.ix.getCol() + 1) * k + std::min((data.ix.getCol() + 1), m)) - range.c_start;
                }

                // If omd already exists for this worker, update the range and data
                if (auto omd = (*res[i])->getObjectMetaDataByLocation(addr)) { 
                    (*res[i])->updateRangeObjectMetaDataByID(omd->omd_id, &range);
                    dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);                    
                }
                else { // else create new omd entry
                    // TODO allocate specific implementation based on alloc_type
                    allocationDescriptor = new AllocationDescriptorDistributedGRPC(
                                                ctx, 
                                                addr,
                                                data);            
                    ((*res[i]))->addObjectMetaData(allocationDescriptor, &range);                    
                } 
            }
        }

        // Find alloc_type
        IAllocationDescriptorDistributed *backend;
        switch (alloc_type){
            case ALLOCATION_TYPE::DIST_GRPC:
                backend = new AllocationDescriptorDistributedGRPC();
                break;
            case ALLOCATION_TYPE::DIST_OPENMPI:
                std::runtime_error("MPI support missing");
                break;
                    
            default:
                std::runtime_error("No distributed implementation found");
                break;
        }         
        auto results = backend->Compute(args, numInputs, mlirCode);
        size_t outputIdx = 0;
        for (auto &output : results){ 
            for (auto &workerResponse : output) {
                auto worker_identifier = workerResponse.first;
                auto resMat = *res[outputIdx];
                auto omd_id = resMat->getObjectMetaDataByLocation(worker_identifier)->omd_id;

                auto storedInfo = workerResponse.second;
                auto omd = resMat->getObjectMetaDataByID(omd_id);
            
                auto data = dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).getDistributedData();
                data.filename = storedInfo.filename;
                data.numRows = storedInfo.numRows;
                data.numCols = storedInfo.numCols;
                data.isPlacedAtWorker = true;
                dynamic_cast<IAllocationDescriptorDistributed&>(*(omd->allocation)).updateDistributedData(data);
            }
            outputIdx++;
        }        
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H