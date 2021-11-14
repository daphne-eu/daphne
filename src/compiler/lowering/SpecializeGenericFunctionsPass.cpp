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
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/PassManager.h"

#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <set>
#include <unordered_map>

using namespace mlir;

namespace {
    bool isFunctionTemplate(FuncOp op) {
        auto unknownTy = daphne::UnknownType::get(op.getContext());
        return llvm::any_of(op.getType().getInputs(),
            [&unknownTy](Type ty) {
              auto matTy = ty.dyn_cast<daphne::MatrixType>();
              return ty == unknownTy || (matTy && matTy.getElementType() == unknownTy);
            });
    }

    std::string uniqueSpecializedFuncName(const std::string &functionName) {
        static unsigned functionUniqueId = 0;
        return functionName + "-" + std::to_string(++functionUniqueId);
    }

    bool callTypesMatchFunctionTypes(FunctionType functionType, TypeRange callTypes) {
        for(auto zipIt : llvm::zip(functionType.getInputs(), callTypes)) {
            auto funcTy = std::get<0>(zipIt);
            auto callTy = std::get<1>(zipIt);
            if(auto funcMatTy = funcTy.dyn_cast<daphne::MatrixType>()) {
                auto callMatTy = callTy.dyn_cast<daphne::MatrixType>();
                if(!callMatTy || funcMatTy.withSameElementType() != callMatTy.withSameElementType()) {
                    return false;
                }
            }
            else if(funcTy != callTy) {
                return false;
            }
        }
        return true;
    }

    std::vector<Type> getSpecializedFuncArgTypes(FunctionType functionType, TypeRange callTypes) {
        auto unknownTy = daphne::UnknownType::get(functionType.getContext());
        std::vector<mlir::Type> specializedTypes;
        for(auto it : llvm::enumerate(llvm::zip(functionType.getInputs(), callTypes))) {
            auto index = it.index();
            auto funcInTy = std::get<0>(it.value());
            auto specializedTy = std::get<1>(it.value());
            if(funcInTy != specializedTy) {
                auto funcMatTy = funcInTy.dyn_cast<daphne::MatrixType>();
                auto specializedMatTy = specializedTy.dyn_cast<daphne::MatrixType>();
                bool isMatchingUnknownMatrix =
                    funcMatTy && specializedMatTy && funcMatTy.getElementType() == unknownTy;
                if(!isMatchingUnknownMatrix && funcInTy != unknownTy) {
                    std::string s;
                    llvm::raw_string_ostream stream(s);
                    stream << "Call to function template with mismatching types for argument " << index
                           << ": Expected type `" << funcInTy << "`, got `" << specializedTy << "`";
                    throw std::runtime_error(stream.str());
                }
                if(specializedMatTy) {
                    // remove size information
                    specializedTy = specializedMatTy.withSameElementType();
                }
            }
            specializedTypes.push_back(specializedTy);
        }
        return specializedTypes;
    }

    /**
     * @brief Set the result types to the types of the function results.
     * @param results The results for which to fix the types
     * @param functionType The function type
     * @return true if changes where made, else false
     */
    bool fixResultTypes(ResultRange results, FunctionType functionType) {
        bool madeChanges = false;
        for(auto it : llvm::zip(results, functionType.getResults())) {
            auto result = std::get<0>(it);
            auto functionResultTy = std::get<1>(it);
            if(result.getType() != functionResultTy) {
                madeChanges = true;
                result.setType(functionResultTy);
            }
        }
        return madeChanges;
    }

    FuncOp inferTypesInFunction(FuncOp function) {
        // Run inference
        mlir::PassManager pm(function->getContext(), "func");
        pm.enableVerifier(false);
        pm.addPass(daphne::createInferencePass({true, true, false, true}));
        if(failed(pm.run(function))) {
            function.emitError() << "could not infer types for a call of function template: " << function.getName();
            return nullptr;
        }
        return function;
    }

    class SpecializeGenericFunctionsPass
        : public PassWrapper<SpecializeGenericFunctionsPass, OperationPass<ModuleOp>> {
        std::unordered_map<std::string, FuncOp> functions;
        std::multimap<std::string, FuncOp> specializedVersions;
        std::set<FuncOp> visited;

        FuncOp createSpecializedFunction(FuncOp templateFunction, TypeRange specializedTypes) {
            OpBuilder builder(templateFunction);
            auto specializedFunc = templateFunction.clone();
            builder.insert(specializedFunc);

            auto uniqueFuncName = uniqueSpecializedFuncName(templateFunction.sym_name().str());
            specializedFunc.setName(uniqueFuncName);
            functions.insert({uniqueFuncName, specializedFunc});
            specializedVersions.insert({templateFunction.sym_name().str(), specializedFunc});

            // change argument types
            specializedFunc
                .setType(builder.getFunctionType(specializedTypes, specializedFunc.getType().getResults()));
            for(auto it : llvm::zip(specializedFunc.getArguments(), specializedTypes)) {
                std::get<0>(it).setType(std::get<1>(it));
            }
            return inferTypesInFunction(specializedFunc);
        }

        void specializeCallsInFunction(FuncOp function) {
            if(visited.count(function)) {
                return;
            }
            visited.insert(function);
            function.walk([&](daphne::GenericCallOp callOp) {
              auto calledFunction = functions[callOp.callee().str()];
              if(isFunctionTemplate(calledFunction)) {
                  // check for existing specialization that matches
                  auto eqIt = specializedVersions.equal_range(calledFunction.sym_name().str());
                  for(auto it = eqIt.first ; it != eqIt.second ; ++it) {
                      auto specializedFunc = it->second;

                      if(callTypesMatchFunctionTypes(specializedFunc.getType(), callOp->getOperandTypes())) {
                          // reuse existing specialized function
                          callOp.calleeAttr(specializedFunc.sym_nameAttr());
                          if(fixResultTypes(callOp->getResults(), specializedFunc.getType())) {
                              inferTypesInFunction(function);
                          }
                          return;
                      }
                  }

                  // Create specialized function
                  auto
                      specializedTypes = getSpecializedFuncArgTypes(calledFunction.getType(), callOp.getOperandTypes());
                  auto specializedFunc = createSpecializedFunction(calledFunction, specializedTypes);

                  callOp.calleeAttr(specializedFunc.sym_nameAttr());
                  if(fixResultTypes(callOp->getResults(), specializedFunc.getType())) {
                      inferTypesInFunction(function);
                  }
                  specializeCallsInFunction(specializedFunc);
              }
              else {
                  specializeCallsInFunction(calledFunction);
              }
            });
        }

    public:
        void runOnOperation() final;
    };
}

void SpecializeGenericFunctionsPass::runOnOperation() {
    auto module = getOperation();

    module.walk([&](FuncOp funcOp) {
      functions.insert({funcOp.sym_name().str(), funcOp});
    });

    std::vector<FuncOp> initialFunctions;
    for(const auto &entry : functions) {
        initialFunctions.push_back(entry.second);
    }
    for(const auto &function : initialFunctions) {
        if(isFunctionTemplate(function) || visited.count(function))
            continue;
        if(!inferTypesInFunction(function)) {
            return signalPassFailure();
        }
        specializeCallsInFunction(function);
    }
    // delete templates
    for(auto f : functions) {
        if(isFunctionTemplate(f.second)) {
            f.second.erase();
        }
    }
}

std::unique_ptr<Pass> daphne::createSpecializeGenericFunctionsPass() {
    return std::make_unique<SpecializeGenericFunctionsPass>();
}
