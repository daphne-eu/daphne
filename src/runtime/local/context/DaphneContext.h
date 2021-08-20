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

#ifndef SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H
#define SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H

#pragma once

#ifdef USE_CUDA
//	#include <runtime/local/context/CUDAContext.h>
#include <vector>
#include <iostream>
class CUDAContext;
#endif

// This macro is intended to be used in kernel function signatures, such that
// we can change the ubiquitous DaphneContext parameter in a single place, if
// required.
#define DCTX(varname) DaphneContext * varname

/**
 * @brief This class carries all kinds of run-time context information.
 * 
 * An instance of this class is passed to every kernel at run-time. It allows
 * the kernel to retrieve information about the run-time environment.
 */
struct DaphneContext {
    // Feel free to extend this class with any kind of run-time information
    // that might be relevant to some kernel. Each kernel can extract the
    // information it requires and does not need to worry about information it
    // does not require.
    // If you need to add a bunch of related information items, please consider
    // creating an individual struct/class for them and adding a single member
    // of that type here, in order to separate concerns and allow a  high-level
    // overview of the context information.
#ifdef USE_CUDA
//    std::vector<std::unique_ptr<CUDAContext>> cuda_contexts;
    std::vector<CUDAContext*> cuda_contexts;
#endif
    // So far, there is no context information.
    
//    explicit DaphneContext(const DaphneUserConfig& config) {
	DaphneContext() {
#ifdef USE_CUDA
//		if(config.use_cuda)
//			for (const auto& dev : config.cuda_devices)
//				cuda_contexts.push_back(CUDAContext::create(dev));
#endif
    }
    
    ~DaphneContext() {
#ifdef USE_CUDA
		std::cout << "desctructing DaphneContext" << std::endl;

//    		for (auto& ctx : cuda_contexts)
//    			ctx->destroy();
//ToDo: use interface for create/destroy
for (auto ctx : cuda_contexts) {
	delete ctx;
}
    		cuda_contexts.clear();
#endif
    }

#ifdef USE_CUDA
	// ToDo: in a multi device setting this should use a find call instead of a direct [] access
	[[nodiscard]] const CUDAContext* getCUDAContext(int dev_id) const { return cuda_contexts[dev_id]; }

#endif
};

#endif //SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H