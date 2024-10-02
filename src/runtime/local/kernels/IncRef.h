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
#include <runtime/local/datastructures/Structure.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct IncRef {
    static void apply(const DTArg *arg, DCTX(ctx)) = delete;
};

template <> struct IncRef<Structure> {
    static void apply(const Structure *arg, DCTX(ctx)) { arg->increaseRefCounter(); }
};

template <> struct IncRef<char> {
    static void apply(const char *arg, DCTX(ctx)) {
        // Increase the reference counter.
        ctx->stringRefCount.inc(arg);
    }
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTArg> void incRef(const DTArg *arg, DCTX(ctx)) { IncRef<DTArg>::apply(arg, ctx); }
