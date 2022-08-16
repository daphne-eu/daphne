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
#include <runtime/local/vectorized/LoadPartitioning.h>
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

using namespace std;
using namespace mlir;
using namespace llvm::cl;

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
    
    // ------------------------------------------------------------------------
    // Define options
    // ------------------------------------------------------------------------
    
    // Option categories ------------------------------------------------------
    
    // TODO We will probably subdivide the options into multiple groups later.
    OptionCategory daphneOptions("DAPHNE Options");

    // Options ----------------------------------------------------------------
    
    // Scheduling options

    opt<SelfSchedulingScheme> taskPartitioningScheme(
            cat(daphneOptions), desc("Choose task partitioning scheme:"),
            values(
                clEnumVal(STATIC , "Static (default)"),
                clEnumVal(SS, "Self-scheduling"),
                clEnumVal(GSS, "Guided self-scheduling"),
                clEnumVal(TSS, "Trapezoid self-scheduling"),
                clEnumVal(FAC2, "Factoring self-scheduling"),
                clEnumVal(TFSS, "Trapezoid Factoring self-scheduling"),
                clEnumVal(FISS, "Fixed-increase self-scheduling"),
                clEnumVal(VISS, "Variable-increase self-scheduling"),
                clEnumVal(PLS, "Performance loop-based self-scheduling"),
                clEnumVal(MSTATIC, "Modified version of Static, i.e., instead of n/p, it uses n/(4*p) where n is number of tasks and p is number of threads"),
                clEnumVal(MFSC, "Modified version of fixed size chunk self-scheduling, i.e., MFSC does not require profiling information as FSC"),
                clEnumVal(PSS, "Probabilistic self-scheduling")
            )
    );

    opt<int> numberOfThreads(
            "num-threads", cat(daphneOptions),
            desc(
                "Define the number of the CPU threads used by the vectorized execution engine "
                "(default is equal to the number of physcial cores on the target node that executes the code)"
            )
    );
    opt<int> minimumTaskSize(
            "grain-size", cat(daphneOptions),
            desc(
                "Define the minimum grain size of a task (default is 1)"
            )
    );
    opt<bool> useVectorizedPipelines(
            "vec", cat(daphneOptions),
            desc("Enable vectorized execution engine")
    );
    
    // Other options
    
    opt<bool> noObjRefMgnt(
            "no-obj-ref-mgnt", cat(daphneOptions),
            desc(
                    "Switch off garbage collection by not managing data "
                    "objects' reference counters"
            )
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
    opt<bool> cuda(
            "cuda", cat(daphneOptions),
            desc("Use CUDA")
    );
    opt<string> libDir(
            "libdir", cat(daphneOptions),
            desc("The directory containing kernel libraries")
    );

    enum ExplainArgs {
      kernels,
      llvm,
      parsing,
      parsing_simplified,
      property_inference,
      sql,
      vectorized,
      obj_ref_mgnt
    };

    llvm::cl::list<ExplainArgs> explainArgList(
        "explain", cat(daphneOptions),
        llvm::cl::desc("Show DaphneIR after certain compiler passes (separate "
                       "multiple values by comma, the order is irrelevant)"),
        llvm::cl::values(
            clEnumVal(parsing, "Show DaphneIR after parsing"),
            clEnumVal(parsing_simplified, "Show DaphneIR after parsing and some simplifications"),
            clEnumVal(sql, "Show DaphneIR after SQL parsing"),
            clEnumVal(property_inference, "Show DaphneIR after property inference"),
            clEnumVal(vectorized, "Show DaphneIR after vectorization"),
            clEnumVal(obj_ref_mgnt, "Show DaphneIR after managing object references"),
            clEnumVal(kernels, "Show DaphneIR after kernel lowering"),
            clEnumVal(llvm, "Show DaphneIR after llvm lowering")),
        CommaSeparated);

    llvm::cl::list<string> scriptArgs1(
            "args", cat(daphneOptions),
            desc(
                    "Alternative way of specifying arguments to the DaphneDSL "
                    "script; must be a comma-separated list of name-value-pairs, "
                    "e.g., `--args x=1,y=2.2`"
            ),
            CommaSeparated
    );
    const std::string configFileInitValue = "-";
    opt<string> configFile(
        "config", cat(daphneOptions),
        desc("A JSON file that contains the DAPHNE configuration"),
        value_desc("filename"),
        llvm::cl::init(configFileInitValue)
    );

    // Positional arguments ---------------------------------------------------
    
    opt<string> inputFile(Positional, desc("script"), Required);
    llvm::cl::list<string> scriptArgs2(ConsumeAfter, desc("[arguments]"));

    // ------------------------------------------------------------------------
    // Parse arguments
    // ------------------------------------------------------------------------
    
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
        if (configFile != configFileInitValue && ConfigParser::fileExists(configFile)) {
            ConfigParser::readUserConfig(configFile, user_config);
        }
    }
    catch(std::exception & e) {
        std::cerr << "Error while reading user config: " << e.what() << std::endl;
        return StatusCode::PARSER_ERROR;
    }
    
//    user_config.debug_llvm = true;
    user_config.use_vectorized_exec = useVectorizedPipelines;
    user_config.use_obj_ref_mgnt = !noObjRefMgnt;
    user_config.libdir = libDir.getValue();
    user_config.library_paths.push_back(user_config.libdir + "/libAllKernels.so");
    user_config.taskPartitioningScheme = taskPartitioningScheme;
    user_config.numberOfThreads = numberOfThreads;
    user_config.minimumTaskSize = minimumTaskSize;

    for (auto explain : explainArgList) {
        switch (explain) {
            case kernels:
                user_config.explain_kernels = true;
                break;
            case llvm:
                user_config.explain_llvm = true;
                break;
            case parsing:
                user_config.explain_parsing = true;
                break;
            case parsing_simplified:
                user_config.explain_parsing_simplified = true;
                break;
            case property_inference:
                user_config.explain_property_inference = true;
                break;
            case sql:
                user_config.explain_sql = true;
                break;
            case vectorized:
                user_config.explain_vectorized = true;
                break;
            case obj_ref_mgnt:
                user_config.explain_obj_ref_mgnt = true;
                break;
        }
    }

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

    // Further, process the module, including optimization and lowering passes.
    try{
        if (!executor.runPasses(moduleOp)) {
            return StatusCode::PASS_ERROR;
        }
    }
    catch(std::exception & e){
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
    }
    catch(std::exception & e){
        std::cerr << "Execution error: " << e.what() << std::endl;
        return StatusCode::EXECUTION_ERROR;
    }

    return StatusCode::SUCCESS;
}
