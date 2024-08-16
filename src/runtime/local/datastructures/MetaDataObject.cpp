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

DataPlacement* MetaDataObject::addDataPlacement(const IAllocationDescriptor *allocInfo, Range *r) {
    data_placements[static_cast<size_t>(allocInfo->getType())].emplace_back(std::make_unique<DataPlacement>(
            allocInfo->clone(), r == nullptr ? nullptr : r->clone()));
    return data_placements[static_cast<size_t>(allocInfo->getType())].back().get();
}

auto MetaDataObject::getDataPlacementByType(ALLOCATION_TYPE type) const
        -> const std::vector<std::unique_ptr<DataPlacement>>* {
    return &(data_placements[static_cast<size_t>(type)]);
}

DataPlacement* MetaDataObject::getDataPlacementByLocation(const std::string& location) const {
    for (const auto &_omdType: data_placements) {
        for (auto &_omd: _omdType) {
            if(_omd->allocation->getLocation() == location)
                return const_cast<DataPlacement *>(_omd.get());
        }
    }
    return nullptr;
}

void MetaDataObject::updateRangeDataPlacementByID(size_t id, Range *r) {
    for(auto &_omdType : data_placements) {
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
        for (auto &_omd: _omdType) {
            if(_omd->dp_id == id)
                return const_cast<DataPlacement *>(_omd.get());
        }
    }
    return nullptr;
}

const DataPlacement* MetaDataObject::findDataPlacementByType(const IAllocationDescriptor *alloc_desc, const Range *range) const {
    auto res = getDataPlacementByType(alloc_desc->getType());
    if(res->empty())
        return nullptr;
    else {
        for (size_t i = 0; i < res->size(); ++i) {
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
