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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/List.h>
#include <runtime/local/datastructures/Structure.h>

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT>
void createList(List<DT> *&res, const DT **elems, size_t numElems, DCTX(ctx)) {
    res = DataObjectFactory::create<List<DT>>();
    for (size_t i = 0; i < numElems; i++)
        res->append(elems[i]);
}
