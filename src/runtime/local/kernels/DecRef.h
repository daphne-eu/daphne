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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct DecRef {
    static void apply(const DTArg *arg, DCTX(ctx)) = delete;
};

template <> struct DecRef<Structure> {
    static void apply(const Structure *arg, DCTX(ctx)) { DataObjectFactory::destroy(arg); }
};

template <> struct DecRef<char> {
    static void apply(const char *arg, DCTX(ctx)) {
        // Decrease the reference counter. If it became zero, delete the string.
        if (!ctx->stringRefCount.dec(arg)) {
            delete[] arg;
        }
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void decRef(const DTArg *arg, DCTX(ctx)) { DecRef<DTArg>::apply(arg, ctx); }
