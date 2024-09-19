/*
 * Copyright 2023 The DAPHNE Consortium
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

#include "CreateDaphneContext.h"
#include "util/KernelDispatchMapping.h"

void createDaphneContext(DaphneContext *&res, uint64_t configPtr,
                         uint64_t dispatchMappingPtr, uint64_t statisticsPtr,
                         uint64_t stringRefCountPtr) {
    auto config = reinterpret_cast<DaphneUserConfig *>(configPtr);
    auto dispatchMapping =
        reinterpret_cast<KernelDispatchMapping *>(dispatchMappingPtr);
    auto statistics = reinterpret_cast<Statistics *>(statisticsPtr);
    auto stringRefCounter =
        reinterpret_cast<StringRefCounter *>(stringRefCountPtr);
    if (config->log_ptr != nullptr)
        config->log_ptr->registerLoggers();
    res = new DaphneContext(*config, *dispatchMapping, *statistics,
                            *stringRefCounter);
}
