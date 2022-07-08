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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ObjectMetaData.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <array>

/**
 * @brief The base class of all data structure implementations.
 */
class Structure
{
private:
    mutable size_t refCounter;
    mutable std::mutex refCounterMutex;
    
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);

protected:
    size_t numRows;
    size_t numCols;

    Structure(size_t numRows, size_t numCols) : refCounter(1), numRows(numRows), numCols(numCols) { // nothing to do
    };

public:
    virtual ~Structure() {
//        for(const auto& _omdType : objectMetaData) {
//            for (auto& _omd: _omdType) {
//                _omd.destroy();
//            }
//        }
    }
    
    size_t getRefCounter() const {
        return refCounter;
    }
    
    /**
     * @brief Increases the reference counter of this data object.
     * 
     * The access is protected by a mutex, such that multiple threads may call
     * this method concurrently.
     */
    void increaseRefCounter() const {
        refCounterMutex.lock();
        refCounter++;
        refCounterMutex.unlock();
    }
    
    // Note that there is no method for decreasing the reference counter here.
    // Instead, use DataObjectFactory::destroy(). It is important that the
    // reference counter becoming zero triggers the deletion of the data
    // object. Thus, we cannot handle it here.

    [[nodiscard]] size_t getNumRows() const
    {
        return numRows;
    }

    [[nodiscard]] size_t getNumCols() const
    {
        return numCols;
    }

    [[nodiscard]] size_t getNumItems() const
    {
        return numRows * numCols;
    }

    /**
     * @brief Prints a human-readable representation of this data object to the
     * given stream.
     * 
     * This method is not optimized for performance. It should only be used for
     * moderately small data objects.
     * 
     * @param os The stream where to print this data object.
     */
    virtual void print(std::ostream & os) const = 0;

    /**
     * @brief Extracts a row range out of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param rl Row range lower bound (inclusive).
     * @param ru Row range upper bound (exclusive).
     * @return 
     */
    virtual Structure* sliceRow(size_t rl, size_t ru) const = 0;

    /**
     * @brief Extracts a column range out of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param cl Column range lower bound (inclusive).
     * @param cu Column range upper bound (exclusive).
     * @return 
     */
    virtual Structure* sliceCol(size_t cl, size_t cu) const = 0;
    
    /**
     * @brief Extracts a rectangular sub-structure (row and column range) out
     * of this structure.
     * 
     * Might be implemented as a zero-copy operation.
     * 
     * @param rl Row range lower bound (inclusive).
     * @param ru Row range upper bound (exclusive).
     * @param cl Column range lower bound (inclusive).
     * @param cu Column range upper bound (exclusive).
     * @return 
     */
    virtual Structure* slice(size_t rl, size_t ru, size_t cl, size_t cu) const = 0;

    ObjectMetaData* addObjectMetaData(const IAllocationDescriptor* allocInfo, Range* r = nullptr) {
        
        objectMetaData[static_cast<size_t>(allocInfo->getType())].emplace_back(std::make_unique<ObjectMetaData>(
                allocInfo->clone(), r == nullptr ? nullptr : r->clone()));
        return objectMetaData[static_cast<size_t>(allocInfo->getType())].back().get();
    }

    const std::vector<std::unique_ptr<ObjectMetaData>>* getObjectMetaDataByType(ALLOCATION_TYPE type) const {
        return &(objectMetaData[static_cast<size_t>(type)]); }
    
    ObjectMetaData* getObjectMetaDataByID(size_t id) const {
        for(const auto& _omdType : objectMetaData) {
            for(auto& _omd : _omdType) {
                if(_omd->omd_id == id)
                    return const_cast<ObjectMetaData*>(_omd.get());
            }
        }
        return nullptr;
    }
    
protected:
    const ObjectMetaData* findObjectMetaData(const IAllocationDescriptor* alloc_desc, const Range* range) const {
        auto res = getObjectMetaDataByType(alloc_desc->getType());
        if(res->empty())
            return nullptr;
        else {
//            for(auto& omd: res) {
//                if(omd.allocation->operator==(_omd.allocation)) {
//                    if((omd.range == nullptr && _omd.range == nullptr) ||
//                       (omd.range != nullptr && omd.range->operator==(_omd.range))) {
//                        return &omd;
//                    }
            for(size_t i = 0; i < res->size(); ++i) {
                if((*res)[i]->allocation->operator==(alloc_desc)) {
                    if(((*res)[i]->range == nullptr && range == nullptr) ||
                       ((*res)[i]->range != nullptr && (*res)[i]->range->operator==(range))) {
                        return (*res)[i].get();
                    }
                }
            }
            return nullptr;
        }
    }

    bool isLatestVersion(size_t omd_id) const {
        return(std::find(latest_version.begin(), latest_version.end(), omd_id) != latest_version.end());
    }

    //    using ObjectMetaDataTypeMap = std::map<uint32_t, ObjectMetaData*>;
//    ObjectMetaDataTypeMap omd;
//    std::vector<ObjectMetaData> objectMetaData;
    std::array<std::vector<std::unique_ptr<ObjectMetaData>>, static_cast<size_t>(ALLOCATION_TYPE::NUM_ALLOC_TYPES)> objectMetaData;

    std::vector<size_t> latest_version;

    // Object Meta Data ID
    size_t omd_id{};
};
