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
#include <compiler/execution/DaphneIrExecutor.h>
#include <parser/config/ConfigParser.h>

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

void parseScriptArgs(const llvm::cl::list<string>& scriptArgsCli, unordered_map<string, string>& scriptArgsFinal) {
    for(const std::string& pair : scriptArgsCli) {
        size_t pos = pair.find('=');
        if(pos == string::npos)
            throw std::runtime_error("script arguments must be specified as name=value, but found '" + pair + "'");
        const string argName = pair.substr(0, pos);
        const string argValue = pair.substr(pos + 1, pair.size());
        if(scriptArgsFinal.count(argName))
            throw runtime_error("script argument: '" + argName + "' was provided more than once");
        scriptArgsFinal.emplace(argName, argValue);
    }
}

int
main(int argc, char** argv)
{
    // ************************************************************************
    // Parse command line arguments
    // ************************************************************************
    
    // Option categories.
    // TODO We will probably subdivide the options into multiple groups later.
    OptionCategory daphneOptions("DAPHNE Options");

    // Options.
    opt<bool> useVectorizedPipelines(
            "vec", cat(daphneOptions),
            desc("Enable vectorized execution engine")
    );
    opt<bool> noFree(
            "no-free", cat(daphneOptions),
            desc("Don't insert FreeOp")
    );
    opt<bool> selectMatrixRepr(
            "select-matrix-repr", cat(daphneOptions),
            desc(
                    "Automatically choose physical matrix representations "
                    "(e.g., dense/sparse)"
            )
    );
    alias selectMatrixReprAlias( // to still support the longer old form
            "select-matrix-representations", aliasopt(selectMatrixRepr),
            desc("Alias for --select-matrix-repr")
    );
    // TODO: parse --explain=[list,of,compiler,passes,to,explain]
    opt<bool> explainKernels(
            "explain-kernels", cat(daphneOptions),
            desc("Show DaphneIR after lowering to kernel calls")
    );
    opt<bool> cuda(
            "cuda", cat(daphneOptions),
            desc("Use CUDA")
    );
    opt<string> libDir(
            "libdir", cat(daphneOptions),
            desc("The directory containing kernel libraries")
    );
    llvm::cl::list<string> scriptArgs1(
            "args", cat(daphneOptions),
            desc(
                    "Alternative way of specifying arguments to the DaphneDSL "
                    "script; must be a comma-separated list of name-value-pairs, "
                    "e.g., `--args x=1,y=2.2`"
            ),
            CommaSeparated
    );
    opt<string> configFile(
        "config", cat(daphneOptions),
        desc("Specify a JSON file that contains the Daphne configuration."),
        value_desc("filename")
    );

    // Positional arguments.
    opt<string> inputFile(Positional, desc("script"), Required);
    llvm::cl::list<string> scriptArgs2(ConsumeAfter, desc("[arguments]"));

    // Parse arguments.
    HideUnrelatedOptions(daphneOptions);
    extrahelp(
            "\nEXAMPLES:\n\n"
            "  daphne example.daphne\n"
            "  daphne --vec example.daphne x=1 y=2.2 z=\"foo\"\n"
            "  daphne --vec --args x=1,y=2.2,z=\"foo\" example.daphne\n"
            "  daphne --vec --args x=1,y=2.2 example.daphne z=\"foo\"\n"
    );
    ParseCommandLineOptions(
            argc, argv,
            "The DAPHNE Prototype.\n\nThis program compiles and executes a DaphneDSL script.\n"
    );
    
    // ************************************************************************
    // Process parsed arguments
    // ************************************************************************
    
    // Initialize user configuration.
    DaphneUserConfig user_config{};
    try {
        if (ConfigParser::fileExists(configFile)) {
            ConfigParser::readUserConfig(configFile, user_config);
        }
    }
    catch(std::exception & e) {
        std::cerr << "Reading user config error: " << e.what() << std::endl;
        return StatusCode::EXECUTION_ERROR;
    }
    
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
    unordered_map<string, string> scriptArgsFinal;
    try {
        parseScriptArgs(scriptArgs2, scriptArgsFinal);
        parseScriptArgs(scriptArgs1, scriptArgsFinal);
    }
    catch(exception& e) {
        std::cerr << "Parser error: " << e.what() << std::endl;
        return StatusCode::PARSER_ERROR;
    }
    
    // ************************************************************************
    // Compile and execute script
    // ************************************************************************

    // Creates an MLIR context and loads the required MLIR dialects.
    DaphneIrExecutor
        executor(std::getenv("DISTRIBUTED_WORKERS"), selectMatrixRepr, user_config);

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
    DaphneDSLParser parser(scriptArgsFinal);
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
