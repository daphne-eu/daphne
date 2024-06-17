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

#include "DataPlacement.h"
#include "MetaDataObject.h"

#include <runtime/local/datastructures/AllocationDescriptorHost.h>

DataPlacement* MetaDataObject::addDataPlacement(std::vector<std::unique_ptr<IAllocationDescriptor>>& allocInfos, Range *r) {
    if(r) {
        auto type = static_cast<size_t>(allocInfos.front()->getType());
        range_data_placements[type].emplace_back(std::make_unique<DataPlacement>(std::move(allocInfos), r->clone()));
        return range_data_placements[type].back().get();
    }
    else {
        auto type = static_cast<size_t>(allocInfos.front()->getType());
        data_placements[type] = std::make_unique<DataPlacement>(
                std::move(allocInfos), nullptr);
        return data_placements[type].get();
    }

}

auto MetaDataObject::getRangeDataPlacementByType(ALLOCATION_TYPE type) const
        -> const std::vector<std::unique_ptr<DataPlacement>>* {
    return &(range_data_placements[static_cast<size_t>(type)]);
}

auto MetaDataObject::getDataPlacementByType(ALLOCATION_TYPE type) const -> DataPlacement* {
    return data_placements[static_cast<size_t>(type)].get();
}

DataPlacement* MetaDataObject::getDataPlacementByLocation(const std::string& location) const {
    // ToDo: no range?
    for (const auto &_omdType: range_data_placements) {
        for (auto &_omd: _omdType) {
            if(_omd->getLocation() == location)
                return const_cast<DataPlacement *>(_omd.get());
        }
    }
    return nullptr;
}

void MetaDataObject::updateRangeDataPlacementByID(size_t id, Range *r) {
    for(auto &_omdType : range_data_placements) {
        for(auto& _omd : _omdType) {
            if(_omd->getID() == id){
                _omd->setRange(r->clone());
                return;
            }
        }
    }
}

DataPlacement *MetaDataObject::getDataPlacementByID(size_t id) const {
    for (const auto &dp: data_placements) {
        if(dp)
            if(dp->getID() == id)
                return const_cast<DataPlacement *>(dp.get());
    }
    for (const auto &rdp_by_type: range_data_placements) {
        for (auto &rdp: rdp_by_type) {
            if(rdp)
                if(rdp->getID() == id)
                    return const_cast<DataPlacement *>(rdp.get());
        }
    }
    return nullptr;
}

bool MetaDataObject::isLatestVersion(size_t placement) const {
    return (std::find(latest_version.begin(), latest_version.end(), placement) != latest_version.end());
}

void MetaDataObject::addLatest(size_t id) {
    latest_version.push_back(id);
}

void MetaDataObject::setLatest(size_t id) {
    latest_version.clear();
    latest_version.push_back(id);
}

auto MetaDataObject::getLatest() const -> std::vector<size_t> {
    return latest_version;
}

DataPlacement *MetaDataObject::getDataPlacement(const IAllocationDescriptor* alloc_desc) {
    auto dp = getDataPlacementByType(alloc_desc->getType());
    if(!dp) {
        // find other allocation type X (preferably host allocation) to transfer from in latest_version
        auto latest = getLatest();
        DataPlacement *placement = getDataPlacementByID(latest.front());

        std::vector<std::unique_ptr<IAllocationDescriptor>> allocations;
        for(uint32_t i = 0; i < placement->getNumAllocations(); ++i) {
            auto other_alloc = placement->getAllocation(i);
            auto new_alloc = alloc_desc->createAllocation(other_alloc->getSize(), false);

            // transfer to requested data placement
            // ToDo: this works for the HOST <-> CUDA use case for now
            // ToDo: update *Matrix own shared ptrs when transferring back from other storage
            if(new_alloc->getType() == ALLOCATION_TYPE::HOST)
                other_alloc->transferFrom(new_alloc->getData().get(), new_alloc->getSize());
            else
                new_alloc->transferTo(other_alloc->getData().get(), other_alloc->getSize());
            allocations.emplace_back(std::move(new_alloc));
        }
        return addDataPlacement(allocations);
    }
    else {
        bool isLatest = isLatestVersion(dp->getID());
        if(!isLatest) {
            auto latest = getDataPlacementByID(getLatest().front());
            for(uint32_t i = 0; i < latest->getNumAllocations(); ++i) {
                auto other_alloc = latest->getAllocation(i);
                dp->getAllocation(i)->transferTo(other_alloc->getData().get(), other_alloc->getSize());
            }
        }
        return dp;
    }
}

std::pair<uint32_t, std::byte*> MetaDataObject::getDataInternal(uint32_t alloc_idx, const IAllocationDescriptor* alloc_desc, const Range* range) {
    DataPlacement *dp;
    if (!range) {
        if (!alloc_desc) {
            auto ad = AllocationDescriptorHost();
            dp = getDataPlacement(&ad);
        } else
            dp = getDataPlacement(alloc_desc);

        return std::make_pair(dp->getID(), dp->getAllocation(alloc_idx)->getData().get());
    } else
        throw std::runtime_error("Range support under construction");
}

const std::byte* MetaDataObject::getData(uint32_t alloc_idx, const IAllocationDescriptor *alloc_desc, const Range *range) const {
    const std::lock_guard<std::mutex> lock(mtx);

    auto [id, ptr] = const_cast<MetaDataObject*>(this)->getDataInternal(alloc_idx, alloc_desc, range);
    const_cast<MetaDataObject*>(this)->addLatest(id);
    return ptr;
}

 std::byte*MetaDataObject::getData(uint32_t alloc_idx, const IAllocationDescriptor *alloc_desc, const Range *range) {
     const std::lock_guard<std::mutex> lock(mtx);

     auto [id, ptr] = getDataInternal(alloc_idx, alloc_desc, range);
     setLatest(id);
     return ptr;
}