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

#ifndef DAPHNE_PROTOTYPE_CUDA_INITCONTEXT_H
#define DAPHNE_PROTOTYPE_CUDA_INITCONTEXT_H

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/CUDAContext.h>

#include <iostream> // remove

// ****************************************************************************
// Convenience function
// ****************************************************************************

static void initCUDAContext(DCTX(ctx)) {
#ifdef NDEBUG
	std::cerr << "ToDo: provide user config to initCUDAContext" << std::endl;
#endif
	ctx->cuda_contexts.emplace_back(CUDAContext::createCudaContext(0));
}

#endif //DAPHNE_PROTOTYPE_CUDA_INITCONTEXT_H
