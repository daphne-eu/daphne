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

#include <api/cli/StatusCode.h>
#include <api/cli/DaphneUserConfig.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include "compiler/execution/DaphneIrExecutor.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#ifdef USE_CUDA
    #include <runtime/local/kernels/CUDA/HostUtils.h>
#endif

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace mlir;

void printHelp(const std::string & cmd) {
    cout << "Usage: " << cmd << " FILE [--args {ARG=VAL}] [--vec] [--select-matrix-representations]" <<
     "[--cuda] [--libdir=<path-to-libs>] [--explain] [--no-free]"<< endl;
}

int
main(int argc, char** argv)
{
    // Parse command line arguments.
    // TODO Rather than implementing this ourselves, we should use some library
    // (see issue #105).
    std::vector<std::string> args(argv, argv + argc);
    string inputFile;
    unordered_map<string, string> scriptArgs;
    bool selectMatrixRepresentations = false;
    DaphneUserConfig user_config{};
//    user_config.debug_llvm = true;
    if(argc < 2) {
        printHelp(args[0]);
        exit(1);
    }
    else {
        if(args[1] == "-h" || args[1] == "--help") {
            printHelp(args[0]);
            exit(0);
        }
        else {
            inputFile = args[1];
            for(int argPos = 2; argPos < argc; argPos++) {
#ifndef NDEBUG
                std::cout << "arg[" << argPos << "]: " << args[argPos] << std::endl;
#endif
                if(args[argPos] == "--args") {
                    int i;
                    for(i = argPos + 1; i < argc; i++) {
                        const std::string pair = args[i];
                        size_t pos_eq = pair.find('=');
						size_t pos_dash = pair.find("--");
                        if(pos_eq == std::string::npos || pos_dash != std::string::npos)
                            break;
                        scriptArgs.emplace(
                            pair.substr(0, pos_eq), // arg name
                            pair.substr(pos_eq + 1, pair.size()) // arg value
                        );
                    }
                    argPos = i - 1;
                }
                else if(args[argPos] == "--vec") {
                    user_config.use_vectorized_exec = true;
                }
                else if(args[argPos] == "--no-free") {
                    user_config.use_freeOps = false;
                }
                else if(args[argPos] == "--explain") {
                    // Todo: parse --explain=[list,of,compiler,passes,to,explain]
                    user_config.explain_kernels = true;
//                    user_config.explain_llvm = true;
                }
                else if(args[argPos] == "--select-matrix-representations") {
                    selectMatrixRepresentations = true;
                }
                else if(args[argPos] == "--cuda") {
                    int device_count = 0;
#ifdef USE_CUDA
                    CHECK_CUDART(cudaGetDeviceCount(&device_count));
#endif
                    if(device_count < 1)
                        std::cerr << "WARNING: CUDA ops requested by user option but no suitable device found" << std::endl;
                    else {
                        user_config.use_cuda = true;
                    }
                }
                else if (std::string(args[argPos]).find("--libdir") != std::string::npos) {
                    const std::string pair = args[argPos];
                    size_t pos = pair.find('=');
                    if(pos == std::string::npos) {
                        std::cerr << "Warning: Malformed parameter " << pair << std::endl;
                        continue;
                    }
                    user_config.libdir = pair.substr(pos + 1, pair.size());
                    user_config.library_paths.push_back(user_config.libdir + "/libAllKernels.so");
                }
                else {
					std::cout << "unknown arg: " << args[argPos] << std::endl;
                    printHelp(args[0]);
                    exit(1);
                }
            }
        }
    }

	// add this after the cli args loop to work around args order
	if(!user_config.libdir.empty() && user_config.use_cuda)
		user_config.library_paths.push_back(user_config.libdir + "/libCUDAKernels.so");

    // Creates an MLIR context and loads the required MLIR dialects.
    DaphneIrExecutor
        executor(std::getenv("DISTRIBUTED_WORKERS"), selectMatrixRepresentations, user_config);

    // Create an OpBuilder and an MLIR module and set the builder's insertion
    // point to the module's body, such that subsequently created DaphneIR
    // operations are inserted into the module.
    OpBuilder builder(executor.getContext());
    auto loc = mlir::FileLineColLoc::get(builder.getIdentifier(inputFile), 0, 0);
    auto moduleOp = ModuleOp::create(loc);
    auto * body = moduleOp.getBody();
    builder.setInsertionPoint(body, body->begin());

    // Parse the input file and generate the corresponding DaphneIR operations
    // inside the module, assuming DaphneDSL as the input format.
    DaphneDSLParser parser(scriptArgs);
    try {
        parser.parseFile(builder, inputFile);
    }
    catch(std::exception & e) {
        std::cerr << "Parser error: " << e.what() << std::endl;
        return StatusCode::PARSER_ERROR;
    }
    
    // Further process the module, including optimization and lowering passes.
    if (!executor.runPasses(moduleOp)) {
        return StatusCode::PASS_ERROR;
    }

    // JIT-compile the module and execute it.
    // module->dump(); // print the LLVM IR representation
    auto engine = executor.createExecutionEngine(moduleOp);
    auto error = engine->invoke("main");
    if (error) {
        llvm::errs() << "JIT-Engine invocation failed: " << error;
        return StatusCode::EXECUTION_ERROR;
    }

    return StatusCode::SUCCESS;
}
