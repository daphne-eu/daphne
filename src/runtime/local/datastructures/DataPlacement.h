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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAPLACEMENT_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAPLACEMENT_H

#include <map>
#include <grpcpp/grpcpp.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <ir/daphneir/Daphne.h>



// ****************************************************************************
// Distributed Runtime
// ****************************************************************************

class DistributedIndex
{
public:
    DistributedIndex() : row_(0), col_(0)
    {}
    DistributedIndex(size_t row, size_t col) : row_(row), col_(col)
    {}

    size_t getRow() const
    {
        return row_;
    }
    size_t getCol() const
    {
        return col_;
    }

    bool operator<(const DistributedIndex rhs) const
    {
        if (row_ < rhs.row_)
            return true;
        else if (row_ == rhs.row_)
            return col_ < rhs.col_;
        return false;
    }

private:
    size_t row_;
    size_t col_;
};

class DistributedData
{
public:
    // Default constructor
    DistributedData() { } ;

    DistributedData(DistributedIndex ix, distributed::StoredData data)
        : ix_(ix), data_(data)
    {}
    DistributedData(distributed::StoredData data)
        : data_(data)
    {}

    const DistributedIndex getDistributedIndex() const
    { return ix_; }
    const distributed::StoredData getData() const
    { return data_; }
    
private:
    DistributedIndex ix_;
    distributed::StoredData data_;

};



// ****************************************************************************
// DataPlacement
// ****************************************************************************

/**
 * @brief Class containing all information and methods regarding metadata
 * for a Structure object (e.g. Where that data is placed, GPUs, distributed workers etc.).
 */
class DataPlacement
{
public:
    using DistributedMap = std::map<std::string, DistributedData>;
private:
    /** Distributed Runtime **/
    DistributedMap distributedMap_;

public:
    /** Default constructor **/
    DataPlacement () { } ;

    // ----------------------------------------------------------------------------
    // Distributed Runtime
    // ----------------------------------------------------------------------------

    DataPlacement (DistributedMap distributedMap) : distributedMap_(distributedMap) 
    { } ;
    // If it is placed on the distributed workers
    bool isPlacedOnWorkers = false;

    // Information on how to collect result
    mlir::daphne::VectorCombine combineType;
    
    const DistributedMap getMap() const
    { return distributedMap_; }
};

#endif //SRC_RUNTIME_LOCAL_DATASTRUCTURES_DATAPLACEMENT_H