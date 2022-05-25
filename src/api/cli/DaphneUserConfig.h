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

#include <runtime/local/vectorized/LoadPartitioning.h>

#include <vector>
#include <string>
#include <memory>
#include <vector>

/*
 * Container to pass around user configuration
 */
struct DaphneUserConfig {
    // Remember to update src/parser/config/UserConfig.json accordingly!

    bool use_cuda = false;
    bool use_oneapi = false;
    bool use_vectorized_exec = false;
    bool use_obj_ref_mgnt = true;
    bool cuda_fuse_any = false;
    bool vectorized_single_queue = false;

    bool debug_llvm = false;
    bool explain_kernels = false;
    bool explain_llvm = false;
    bool explain_parsing = false;
    bool explain_property_inference = false;
    bool explain_sql = false;
    bool explain_vectorized = false;
    bool explain_obj_ref_mgnt = false;
    SelfSchedulingScheme taskPartitioningScheme = STATIC;
    int numberOfThreads = -1;
    int minimumTaskSize = 1;
    
#ifdef USE_CUDA
    // User config holds once context atm for convenience until we have proper system infrastructure

    // CUDA device IDs (future work, as we create only one context atm)
    std::vector<int> cuda_devices;

    // ToDo: This is an arbitrary default taken from sample code
//    int cublas_workspace_size = 1024 * 1024 * 4;
#endif
    std::string libdir;
    std::vector<std::string> library_paths;
};
