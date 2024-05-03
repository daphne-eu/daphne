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

#pragma once

class DataPlacement;
#include "IAllocationDescriptor.h"
#include "Range.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <vector>

/**
 * @brief The MetaDataObject class contains meta data of a data structure (Frame, Matrix)
 *
 * The MetaDataObject holds a vector of data placements (separated by type) and a vector of IDs of
 * data placements that all hold the current/latest version of the contained data.
 * Additionaly, this class contains methods to access/manipulate the contained information.
 */
class MetaDataObject {
    std::array<std::vector<std::unique_ptr<DataPlacement>>,
            static_cast<size_t>(ALLOCATION_TYPE::NUM_ALLOC_TYPES)> data_placements;
    std::vector<size_t> latest_version;

public:
    DataPlacement *addDataPlacement(const IAllocationDescriptor *allocInfo, Range *r = nullptr);
    const DataPlacement *findDataPlacementByType(const IAllocationDescriptor *alloc_desc, const Range *range) const;
    [[nodiscard]] DataPlacement *getDataPlacementByID(size_t id) const;
    [[nodiscard]] DataPlacement *getDataPlacementByLocation(const std::string& location) const;
    [[nodiscard]] auto getDataPlacementByType(ALLOCATION_TYPE type) const ->
            const std::vector<std::unique_ptr<DataPlacement>>*;
    void updateRangeDataPlacementByID(size_t id, Range *r);

    [[nodiscard]] bool isLatestVersion(size_t placement) const;
    void addLatest(size_t id);
    void setLatest(size_t id);
    [[nodiscard]] auto getLatest() const -> std::vector<size_t>;

};
