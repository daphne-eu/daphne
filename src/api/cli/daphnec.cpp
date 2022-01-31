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
    #include "runtime/local/kernels/CUDA_HostUtils.h"
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
    cout << "Usage: " << cmd << " FILE [--args {ARG=VAL}] [--vec] [--select-matrix-representations]" << endl;
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
    bool useVectorizedPipelines = false;
    bool selectMatrixRepresentations = false;
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
                if(args[argPos] == "--args") {
                    int i;
                    for(i = argPos + 1; i < argc; i++) {
                        const std::string pair = args[i];
                        size_t pos = pair.find('=');
                        if(pos == std::string::npos)
                            break;
                        scriptArgs.emplace(
                            pair.substr(0, pos), // arg name
                            pair.substr(pos + 1, pair.size()) // arg value
                        );
                    }
                    argPos = i - 1;
                }
                else if(args[argPos] == "--vec") {
                    useVectorizedPipelines = true;
                }
                else if(args[argPos] == "--select-matrix-representations") {
                    selectMatrixRepresentations = true;
                }
                else {
                    printHelp(args[0]);
                    exit(1);
                }
            }
        }
    }

    // TODO "libdir" and "cuda" should not be script arguments. Script
    // arguments are those that can be used from within the DaphneDSL script.
    // However, these two arguments are only required during compilation and
    // runtime, users do not need to access them in a script.

    DaphneUserConfig user_config;
    auto it = scriptArgs.find("libdir");
    if(it != scriptArgs.end()) {
        user_config.libdir = it->second;
        user_config.library_paths.push_back(user_config.libdir + "/libAllKernels.so");
    }

#ifdef USE_CUDA
    it = scriptArgs.find("cuda");
    if(it != scriptArgs.end()) {
        if(it->second == "1") {
//            std::cout << "-cuda flag provided" << std::endl;
            int device_count;
              CHECK_CUDART(cudaGetDeviceCount(&device_count));
              if(device_count < 1)
                  std::cerr << "WARNING: CUDA ops requested by user option but no suitable device found" << std::endl;
            else { // NOLINT(readability-misleading-indentation)
                std::cout << "Available CUDA devices: " << device_count << std::endl;
                user_config.use_cuda = true;
            }
        }
    }

    it = scriptArgs.find("libdir");
    if(it != scriptArgs.end()) {
        user_config.library_paths.push_back(user_config.libdir + "/libCUDAKernels.so");
    }
#endif

    // Creates an MLIR context and loads the required MLIR dialects.
    DaphneIrExecutor
        executor(std::getenv("DISTRIBUTED_WORKERS"), useVectorizedPipelines, selectMatrixRepresentations, true, user_config);

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
    try{
        if (!executor.runPasses(moduleOp)) {
            return StatusCode::PASS_ERROR;
        }
    }catch(std::exception & e){
        std::cerr << "Pass error: " << e.what() << std::endl;
        return StatusCode::PASS_ERROR;
    }

    // JIT-compile the module and execute it.
    // module->dump(); // print the LLVM IR representation
    try{
        auto engine = executor.createExecutionEngine(moduleOp);
        auto error = engine->invoke("main");
        if (error) {
            llvm::errs() << "JIT-Engine invocation failed: " << error;
            return StatusCode::EXECUTION_ERROR;
        }
    }catch(std::exception & e){
        std::cerr << "Execution error: " << e.what() << std::endl;
        return StatusCode::EXECUTION_ERROR;
    }

    return StatusCode::SUCCESS;
}
