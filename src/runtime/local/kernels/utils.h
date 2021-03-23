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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_UTILS_H
#define SRC_RUNTIME_LOCAL_KERNELS_UTILS_H

#include <cassert>

/**
 * `typeNew` should be a pointer type.
 */
#define dynamic_cast_assert(typeNew, varNew, varOld) \
    typeNew varNew = dynamic_cast<typeNew>(varOld); \
    assert(varNew && "'" #varOld "'does not have the expected type '" #typeNew "'");

#endif //SRC_RUNTIME_LOCAL_KERNELS_UTILS_H