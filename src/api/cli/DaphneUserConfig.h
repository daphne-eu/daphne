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

#include <api/daphnelib/DaphneLibResult.h>
#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <util/LogConfig.h>
#include <util/DaphneLogger.h>
class DaphneLogger;

#include <vector>
#include <string>
#include <memory>
#include <map>

/*
 * Container to pass around user configuration
 */
struct DaphneUserConfig {
    // Remember to update UserConfig.json accordingly!

    bool use_cuda = false;
    bool use_vectorized_exec = false;
    bool use_distributed = false;
    bool use_obj_ref_mgnt = true;
    bool use_ipa_const_propa = true;
    bool use_phy_op_selection = true;
    bool cuda_fuse_any = false;
    bool vectorized_single_queue = false;
    bool prePartitionRows = false;
    bool pinWorkers = false;
    bool hyperthreadingEnabled = false;
    bool debugMultiThreading = false;
    bool use_fpgaopencl = false;
    bool enable_profiling = false;

    bool debug_llvm = false;
    bool explain_kernels = false;
    bool explain_llvm = false;
    bool explain_parsing = false;
    bool explain_parsing_simplified = false;
    bool explain_property_inference = false;
    bool explain_select_matrix_repr = false;
    bool explain_sql = false;
    bool explain_phy_op_selection = false;
    bool explain_type_adaptation = false;
    bool explain_vectorized = false;
    bool explain_obj_ref_mgnt = false;
    SelfSchedulingScheme taskPartitioningScheme = STATIC;
    QueueTypeOption queueSetupScheme = CENTRALIZED;
	VictimSelectionLogic victimSelection = SEQPRI;
    ALLOCATION_TYPE distributedBackEndSetup= ALLOCATION_TYPE::DIST_MPI; // default value
    int numberOfThreads = -1;
    int minimumTaskSize = 1;
    // minimum considered log level (e.g., no logging below INFO (essentially suppressing DEBUG and TRACE)
    spdlog::level::level_enum log_level_limit = spdlog::level::off;
    std::vector<LogConfig> loggers;
    DaphneLogger* log_ptr{};
    
#ifdef USE_CUDA
    // User config holds once context atm for convenience until we have proper system infrastructure

    // CUDA device IDs (future work, as we create only one context atm)
    std::vector<int> cuda_devices;

    // ToDo: This is an arbitrary default taken from sample code
//    int cublas_workspace_size = 1024 * 1024 * 4;
#endif
#ifdef USE_FPGAOPENCL
    std::vector<int> fpga_devices;
#endif
    
    
    std::string libdir;
    std::vector<std::string> library_paths;
    std::map<std::string, std::vector<std::string>> daphnedsl_import_paths;


    // TODO Maybe the DaphneLib result should better reside in the DaphneContext,
    // but having it here is simpler for now.
    DaphneLibResult* result_struct = nullptr;
};
