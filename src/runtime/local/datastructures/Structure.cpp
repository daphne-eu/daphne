/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/DataPlacement.h>

Structure::Structure(size_t numRows, size_t numCols) : refCounter(1), numRows(numRows), numCols(numCols) {
    mdo = std::make_shared<MetaDataObject>();
};

void Structure::clone_mdo(const Structure* src) {
    // FIXME: This clones the meta data to avoid locking (thread synchronization for data copy)
    for(int i = 0; i < static_cast<int>(ALLOCATION_TYPE::NUM_ALLOC_TYPES); i++) {
        auto placements = src->mdo->getDataPlacementByType(static_cast<ALLOCATION_TYPE>(i));
        for(auto it = placements->begin(); it != placements->end(); it++) {
            auto src_alloc = it->get()->allocation.get();
            auto src_range = it->get()->range.get();
            auto new_data_placement = this->mdo->addDataPlacement(src_alloc, src_range);
            if(src->mdo->isLatestVersion(it->get()->dp_id))
                this->mdo->addLatest(new_data_placement->dp_id);
        }
    }
}