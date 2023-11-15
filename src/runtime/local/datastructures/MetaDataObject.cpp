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

#include "MetaDataObject.h"

DataPlacement* MetaDataObject::addDataPlacement(const IAllocationDescriptor *allocInfo, Range *r) {
    if(r) {
        range_data_placements[static_cast<size_t>(allocInfo->getType())].emplace_back(std::make_unique<DataPlacement>(
                allocInfo->clone(), r->clone()));
        return range_data_placements[static_cast<size_t>(allocInfo->getType())].back().get();
    }
    else {
        data_placements[static_cast<size_t>(allocInfo->getType())] = std::make_unique<DataPlacement>(
                allocInfo->clone(), nullptr);
        return data_placements[static_cast<size_t>(allocInfo->getType())].get();
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
            if(_omd->allocation->getLocation() == location)
                return const_cast<DataPlacement *>(_omd.get());
        }
    }
    return nullptr;
}

void MetaDataObject::updateRangeDataPlacementByID(size_t id, Range *r) {
    for(auto &_omdType : range_data_placements) {
        for(auto& _omd : _omdType) {
            if(_omd->dp_id == id){
                _omd->range = r->clone();
                return;
            }
        }
    }
}

DataPlacement *MetaDataObject::getDataPlacementByID(size_t id) const {
    for (const auto &_omdType: data_placements) {
        if(_omdType->dp_id == id)
            return const_cast<DataPlacement *>(_omdType.get());
    }
    for (const auto &_omdType: range_data_placements) {
        for (auto &_omd: _omdType) {
            if(_omd->dp_id == id)
                return const_cast<DataPlacement *>(_omd.get());
        }
    }
    return nullptr;
}

const DataPlacement* MetaDataObject::findDataPlacementByType(const IAllocationDescriptor *alloc_desc, const Range *range) const {
    if(range) {
        auto res = getRangeDataPlacementByType(alloc_desc->getType());
        if (res->empty())
            return nullptr;
        else {
            for (size_t i = 0; i < res->size(); ++i) {
                if ((*res)[i]->allocation->operator==(alloc_desc)) {
                    if (((*res)[i]->range == nullptr && range == nullptr) ||
                        ((*res)[i]->range != nullptr && (*res)[i]->range->operator==(range))) {
                        return (*res)[i].get();
                    }
                }
            }
            return nullptr;
        }

    }
    else
        return getDataPlacementByType(alloc_desc->getType());
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
    // tuple content: <is latest, latest-id, ptr-to-data-placement>
    std::tuple<bool, size_t, std::byte *> result = std::make_tuple(false, 0, nullptr);
    auto dp = getDataPlacementByType(alloc_desc->getType());
    if(!dp) {
        // find other allocation type X (preferably host allocation) to transfer from in latest_version
        auto latest = getLatest();
        DataPlacement *placement = getDataPlacementByID(latest.front());
//        for (auto &placement_id: latest) {
//            placement = getDataPlacementByID(placement_id);
//            std::get<0>(result) = true;
//            std::get<1>(result) = placement->dp_id;
//            // prefer host allocation
//            if(placement->allocation->getType() == ALLOCATION_TYPE::HOST) {
//                std::get<2>(result) = reinterpret_cast<ValueType *>(values.get());
//                break;
//            }
//        }

        // if we found a data placement that is not in host memory, transfer it there before returning
//        if(std::get<0>(result) == true && std::get<2>(result) == nullptr) {
//            AllocationDescriptorHost myHostAllocInfo;
//            if(!values)
//                this->alloc_shared_values();
//            this->mdo->addDataPlacement(&myHostAllocInfo);
//            placement->allocation->transferFrom(reinterpret_cast<std::byte *>(startAddress()), getBufferSize());
//            std::get<2>(result) = startAddress();
//        }

        // create new data placement
        auto new_data_placement = addDataPlacement(alloc_desc);
        new_data_placement->allocation->createAllocation(placement->allocation->getSize(), false);

        // transfer to requested data placement
        new_data_placement->allocation->transferTo(placement->allocation->getData().get(), placement->allocation->getSize());
//        return std::make_tuple(true, new_data_placement->dp_id, new_data_placement->allocation->getData().get());
        return new_data_placement;
    }
    else {
        bool isLatest = isLatestVersion(dp->dp_id);
        if(!isLatest) {
            auto latest = getDataPlacementByID(getLatest().front());
            dp->allocation->transferTo(latest->allocation->getData().get(), latest->allocation->getSize());
        }
//        return std::make_tuple(latest, dp->dp_id, dp->allocation->getData().get());
        return dp;
    }
}
