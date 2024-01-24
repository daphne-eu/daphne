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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/Range.h>

#include <vector>
#include <string>
#include <cassert>
#include <cstddef>
#include <type_traits>

using mlir::daphne::VectorCombine;


enum class DistributionSchema{
    DISTRIBUTE = 1,
    BROADCAST = 2
};

template<class DT, class ALLOCATOR>
class LoadPartitioningDistributed {
private:
    DistributionSchema distrschema;
    DT *mat;
    std::vector<std::string> workerList;
    size_t taskIndex = 0;
    size_t totalTasks;
    DaphneContext *dctx;

public:

    LoadPartitioningDistributed(DistributionSchema schema, DT *&mat, DCTX(dctx)) :
        distrschema(schema),
        mat(mat),
        dctx(dctx)
    {
        auto ctx = DistributedContext::get(dctx);
        workerList = ctx->getWorkers();
        totalTasks = workerList.size();
    };

    bool HasNextChunk(){
        return taskIndex < totalTasks;
    };

    // Each allocation descriptor might use a different constructor.
    // Here we provide the different implementations.
    // Another solution would be to make sure that every constructor is similar so this would not be needed.
    static ALLOCATOR CreateAllocatorDescriptor(DaphneContext* ctx, const std::string &addr, const DistributedData &data) {
        if constexpr (std::is_same_v<ALLOCATOR, AllocationDescriptorMPI>)
            return AllocationDescriptorMPI(std::stoi(addr), ctx, data);
        else if constexpr (std::is_same_v<ALLOCATOR, AllocationDescriptorGRPC>)
            return AllocationDescriptorGRPC(ctx, addr, data);
        else
            throw std::runtime_error("Unknown allocation type");
    }

    // Set ranges
    Range CreateRange() {
        switch (distrschema) {
            case DistributionSchema::DISTRIBUTE: {
                // Todo support for different distribution schemas
                auto k = mat->getNumRows() / workerList.size();
                auto m = mat->getNumRows() % workerList.size();
                return Range(
                    (taskIndex * k) + std::min(taskIndex, m),
                    0,
                    ((taskIndex + 1) * k + std::min(taskIndex + 1, m)) - ((taskIndex * k) + std::min(taskIndex, m)),
                    mat->getNumCols()
                );
                break;
            }
            case DistributionSchema::BROADCAST:
                return Range(
                    0,
                    0,
                    mat->getNumRows(),
                    mat->getNumCols()
                );
                break;
            default:
                throw std::runtime_error("Unknown distribution scheme");
        }
    }

    // Update current distributed index object based on distribution schema
    DistributedIndex GetDistributedIndex() {
        switch (distrschema) {
            case DistributionSchema::DISTRIBUTE:
                // Todo support for different distribution schemas
                return DistributedIndex(taskIndex, 0);
            case DistributionSchema::BROADCAST:
                return DistributedIndex(0, 0);
            default:
                throw std::runtime_error("Unknown distribution scheme");
        }
    }

    DataPlacement * GetNextChunk(){

        auto workerAddr = workerList.at(taskIndex);

        auto range = CreateRange();

        DataPlacement *dp;
        if ((dp = mat->getMetaDataObject()->getDataPlacementByLocation(workerAddr))) {
            auto data = dynamic_cast<ALLOCATOR&>(*(dp->allocation)).getDistributedData();

            // Check if existing placement matches the same ranges we currently need
            if (data.isPlacedAtWorker) {
                auto existingRange = dp->range.get();
                if (*existingRange == range)
                    data.isPlacedAtWorker = true;
                else {
                    mat->getMetaDataObject()->updateRangeDataPlacementByID(dp->dp_id, &range);
                    data.isPlacedAtWorker = false;
                }
            } else
                mat->getMetaDataObject()->updateRangeDataPlacementByID(dp->dp_id, &range);
            // TODO Currently we do not support distributing/splitting
            // by columns. When we do, this should be changed (e.g. Index(0, taskIndex))
            // This can be decided based on DistributionSchema
            data.ix = GetDistributedIndex();
            dynamic_cast<ALLOCATOR&>(*(dp->allocation)).updateDistributedData(data);
        }
        else { // Else, create new object metadata entry
            DistributedData data;
            // TODO Currently we do not support distributing/splitting
            // by columns. When we do, this should be changed (e.g. Index(0, taskIndex))
            data.ix = GetDistributedIndex();
            auto allocationDescriptor = CreateAllocatorDescriptor(dctx, workerAddr, data);
            dp = mat->getMetaDataObject()->addDataPlacement(&allocationDescriptor, &range);
        }
        taskIndex++;
        return dp;
    }

    static void SetOutputsMetadata(DT **&outputs, size_t numOutputs, VectorCombine *&vectorCombine, DCTX(dctx)) {
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
        // Initialize Distributed index array, needed for results
        std::vector<DistributedIndex> ix(numOutputs, DistributedIndex(0, 0));
        for (auto workerAddr : workers) {
            for (size_t i = 0; i < numOutputs; i++)
            {
                // Get Result ranges
                // TODO Seperate this into a different function and implement different strategies
                auto combineType = vectorCombine[i];
                auto workersSize = workers.size();
                size_t k = 0, m = 0;
                if (combineType == VectorCombine::ROWS)
                {
                    k = (*outputs[i])->getNumRows() / workersSize;
                    m = (*outputs[i])->getNumRows() % workersSize;
                }
                else if (combineType == VectorCombine::COLS)
                {
                    k = (*outputs[i])->getNumCols() / workersSize;
                    m = (*outputs[i])->getNumCols() % workersSize;
                }
                else if (combineType == VectorCombine::ADD)
                {
                    k = (*outputs[i])->getNumCols() / workersSize;
                    m = (*outputs[i])->getNumCols() % workersSize;
                }
                else
                    assert(!"Only Rows/Cols combineType supported atm");

                DistributedData data;
                data.ix = ix[i];
                data.vectorCombine = vectorCombine[i];
                data.isPlacedAtWorker = true;

                // Update distributed index for next iteration
                // and set ranges for objmetadata
                Range range;
                if (vectorCombine[i] == VectorCombine::ROWS)
                {
                    ix[i] = DistributedIndex(ix[i].getRow() + 1, ix[i].getCol());

                    range.r_start = data.ix.getRow() * k + std::min(data.ix.getRow(), m);
                    range.r_len = ((data.ix.getRow() + 1) * k + std::min((data.ix.getRow() + 1), m)) - range.r_start;
                    range.c_start = 0;
                    range.c_len = (*outputs[i])->getNumCols();
                }
                if (vectorCombine[i] == VectorCombine::COLS || vectorCombine[i] == VectorCombine::ADD)
                {
                    ix[i] = DistributedIndex(ix[i].getRow(), ix[i].getCol() + 1);

                    range.r_start = 0;
                    range.r_len = (*outputs[i])->getNumRows();
                    range.c_start = data.ix.getCol() * k + std::min(data.ix.getCol(), m);
                    range.c_len = ((data.ix.getCol() + 1) * k + std::min((data.ix.getCol() + 1), m)) - range.c_start;
                }

                // If dp already exists for this worker, update the range and data
                if (auto dp = (*outputs[i])->getMetaDataObject()->getDataPlacementByLocation(workerAddr))
                {
                    (*outputs[i])->getMetaDataObject()->updateRangeDataPlacementByID(dp->dp_id, &range);
                    dynamic_cast<ALLOCATOR&>(*(dp->allocation)).updateDistributedData(data);
                }
                else
                { // else create new dp entry
                    auto allocationDescriptor = CreateAllocatorDescriptor(dctx, workerAddr, data);
                    ((*outputs[i]))->getMetaDataObject()->addDataPlacement(&allocationDescriptor, &range);
                }
            }
        }
    }
};