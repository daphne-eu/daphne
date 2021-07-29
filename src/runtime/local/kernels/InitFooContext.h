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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_INITFOOCONTEXT_H
#define SRC_RUNTIME_LOCAL_KERNELS_INITFOOCONTEXT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/FooContext.h>

#include <iostream> // remove

// ****************************************************************************
// Convenience function
// ****************************************************************************

void initFooContext(DCTX(ctx)) {
    std::cerr << "initFooContext beg" << std::endl;
    ctx->fooCtx = new FooContext();
    std::cerr << "initFooContext end" << std::endl;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_INITFOOCONTEXT_H