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
#include <parser/daphnesql/DaphneSQLParser.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include "compiler/execution/DaphneIrExecutor.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include <iostream>
#include <memory>

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace mlir;

int
main(int argc, char** argv)
{
    if (argc != 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        cout << "Usage: " << argv[0] << " FILE" << endl;
        exit(1);
    }

    // Parse command line arguments.
    string inputFile(argv[1]);

    // Creates an MLIR context and loads the required MLIR dialects.
    DaphneIrExecutor executor(std::getenv("DISTRIBUTED_WORKERS"));

    // Create an OpBuilder and an MLIR module and set the builder's insertion
    // point to the module's body, such that subsequently created DaphneIR
    // operations are inserted into the module.
    OpBuilder builder(executor.getContext());
    auto moduleOp = ModuleOp::create(builder.getUnknownLoc());
    auto * body = moduleOp.getBody();
    builder.setInsertionPoint(body, body->begin());

    // Parse the input file and generate the corresponding DaphneIR operations
    // inside the module, assuming DaphneDSL as the input format.
    DaphneDSLParser parser;
    std::cout << "Error before here" << std::endl;
    parser.parseFile(builder, inputFile);
    moduleOp->dump();
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

    return 0;
}
