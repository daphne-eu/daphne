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
#include "llvm/Support/CommandLine.h"

#ifdef USE_CUDA
    #include <runtime/local/kernels/CUDA/HostUtils.h>
#endif

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;
using namespace mlir;
using namespace llvm::cl;

void printHelp(const std::string & cmd) {
    cout << "Usage: " << cmd << " FILE [--args {ARG=VAL}] [--vec] [--select-matrix-representations]" <<
     "[--cuda] [--libdir=<path-to-libs>] [--explain] [--no-free]"<< endl;
}
void tokenize(string const &in, const char delim, vector<std::string> &out)
{
    size_t start;
    size_t end = 0;
    while ((start = in.find_first_not_of(delim, end)) != string::npos)
    {
        end = in.find(delim, start);
        out.push_back(in.substr(start, end - start));
    }
}

int
main(int argc, char** argv)
{
    // ************************************************************************
    // Parse command line arguments
    // ************************************************************************
    
    OptionCategory daphneOptions("daphne Options", "Options for controlling the compilation process.");

    opt<bool> useVectorizedPipelines("vec", desc("force tiled execution engine"), cat(daphneOptions));
    opt<bool> noFree("no-free", desc("don't insert FreeOp"), cat(daphneOptions));
    opt<bool> selectMatrixRepresentations("select-matrix-representations", desc("force sparce matrices"),  cat(daphneOptions));
    alias selectMatrixRepresentationsAlias("s", desc("Alias for -select-matrix-representations"), aliasopt(selectMatrixRepresentations));
    // TODO: parse --explain=[list,of,compiler,passes,to,explain]
    opt<bool> explainKernels("explain-kernels", desc("show IR after lowering to kernel calls"),  cat(daphneOptions));
    opt<bool> cuda("cuda", desc("use CUDA"),  cat(daphneOptions));
    opt<string> libDir("libdir", desc("the directory containing kernel libraries"),  cat(daphneOptions));
    opt<string> userArgs("args", desc("user arguments to the daphne script <token1=value,token2=value,....>"),  cat(daphneOptions));

    opt<string> inputFile(Positional, desc("<input file>"), Required);

    HideUnrelatedOptions( daphneOptions);
    ParseCommandLineOptions(argc, argv, " daphne compiler \n\nThis program compiles a daphne script...\n");
    
    // ************************************************************************
    // Process parsed arguments
    // ************************************************************************
    
    // Initialize user configuration.
    
    DaphneUserConfig user_config{};
    
//    user_config.debug_llvm = true;
    user_config.use_vectorized_exec = useVectorizedPipelines;
    user_config.use_freeOps = !noFree;
    user_config.explain_kernels = explainKernels;
    user_config.libdir = libDir;
    user_config.library_paths.push_back(user_config.libdir + "/libAllKernels.so");
    
    if(cuda) {
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

    // add this after the cli args loop to work around args order
    if(!user_config.libdir.empty() && user_config.use_cuda)
            user_config.library_paths.push_back(user_config.libdir + "/libCUDAKernels.so");

    // Extract script args.
    
    unordered_map<string, string> scriptArgs;
    vector<string> tokenArgs;
    tokenize(userArgs.c_str(), ',',tokenArgs);
    for(size_t i = 0; i < tokenArgs.size(); i++){
        const string pair = tokenArgs.at(i);
        size_t pos = pair.find('=');
        if(pos == string::npos)
            break;
        scriptArgs.emplace(
                pair.substr(0, pos), // arg name
                pair.substr(pos + 1, pair.size()) // arg value
        );
    }
    
    // ************************************************************************
    // Compile and execute script
    // ************************************************************************

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
