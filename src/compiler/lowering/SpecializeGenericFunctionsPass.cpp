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
#include <stdexcept>
#include <string>
#include <set>
#include <unordered_map>

using namespace mlir;

namespace {
    /**
     * @brief Checks if the function is a template, by checking the types of input arguments.
     * @param op The `FuncOp` to check
     * @return true if `FuncOp` is a template, false otherwise
     */
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

    /**
     * @brief Check if a function with the given input/output types can be called with the input types given.
     * @param functionType The type of the function
     * @param callTypes The types used in the call
     * @return true if the types match for a call, false otherwise
     */
    bool callTypesMatchFunctionTypes(FunctionType functionType, TypeRange callTypes) {
        for(auto zipIt : llvm::zip(functionType.getInputs(), callTypes)) {
            auto funcTy = std::get<0>(zipIt);
            auto callTy = std::get<1>(zipIt);
            if(auto funcMatTy = funcTy.dyn_cast<daphne::MatrixType>()) {
                auto callMatTy = callTy.dyn_cast<daphne::MatrixType>();
                // Check without shape information
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

    /**
     * @brief Get argument types for the specialized version of a template function.
     * @param functionType The types of the template function.
     * @param callTypes The types used in the call to the specialized version.
     * @return The argument types to use for the specialized version
     */
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

    /**
     * @brief Run partial type and label inference on the given `FuncOp`.
     * @param function The `FuncOp`
     * @return The inferred `FuncOp` (same as input), or `nullptr` if an error happened
     */
    FuncOp inferTypesInFunction(FuncOp function) {
        // Run inference
        mlir::PassManager pm(function->getContext(), "func");
        pm.enableVerifier(false);
        pm.addPass(daphne::createInferencePass({true, true, false, true, false}));
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

        /**
         * @brief Create a specialized version of the template function.
         * @param templateFunction The template function.
         * @param specializedTypes The specialized function arguments
         * @return The specialized function
         */
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

        /**
         * @brief Try to reuse an existing specialization for the given template function
         * @param operandTypes Operand types of the operation
         * @param templateFunction The template function called by the operation
         * @return either an existing and matching `FuncOp`, `nullptr` otherwise
         */
        FuncOp tryReuseExistingSpecialization(TypeRange operandTypes, FuncOp templateFunction) {
            auto eqIt = specializedVersions.equal_range(templateFunction.sym_name().str());
            for(auto it = eqIt.first ; it != eqIt.second ; ++it) {
                auto specializedFunc = it->second;

                if(callTypesMatchFunctionTypes(specializedFunc.getType(), operandTypes)) {
                    // reuse existing specialized function
                    return specializedFunc;
                }
            }
            return nullptr;
        }

        /**
         * @brief Try to reuse an existing specializtion if one exists, else creates a new 
         *  specialization
         * @param operandTypes Operand types of the operation
         * @param calledFunction The function called by the operation
         * @return A `FuncOp`for the specialization
         */
        FuncOp createOrReuseSpecialization(TypeRange operandTypes, FuncOp calledFunction) {
            // check for existing specialization that matches
            FuncOp specializedFunc = tryReuseExistingSpecialization(operandTypes, calledFunction);
            if(!specializedFunc) {
                // Create specialized function
                auto specializedTypes =
                    getSpecializedFuncArgTypes(calledFunction.getType(), operandTypes);
                specializedFunc = createSpecializedFunction(calledFunction, specializedTypes);
            }
            return specializedFunc;
        }

        /**
         * @brief Recursively specializes all functions within a `FuncOp` based on calls to the functions
         * @param function The `FuncOp` to scan for function specializations
         */
        void specializeCallsInFunction(FuncOp function) {
            if(visited.count(function)) {
                return;
            }
            visited.insert(function);
            // Specialize all functions called directly
            function.walk([&](daphne::GenericCallOp callOp) {
                auto calledFunction = functions[callOp.callee().str()];
                if(isFunctionTemplate(calledFunction)) {
                    FuncOp specializedFunc = createOrReuseSpecialization(callOp.getOperandTypes(), calledFunction);
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

            // Specialize all functions called by MapOp
            function.walk([&](daphne::MapOp mapOp) {
                auto calledFunction = functions[mapOp.func().str()];
                if(isFunctionTemplate(calledFunction)) {
                     // Get the element type of the matrix the function should be mapped on
                    mlir::Type opTy = mapOp.arg().getType();
                    auto inpMatrixTy = opTy.dyn_cast<daphne::MatrixType>();
                    FuncOp specializedFunc = createOrReuseSpecialization(inpMatrixTy.getElementType(), calledFunction);
                    mapOp.funcAttr(specializedFunc.sym_nameAttr());
 
                    // We only allow functions that return exactly one result for mapOp
                    if(specializedFunc.getType().getNumResults() != 1) 
                        throw std::runtime_error(
                            "map expects a function with exactly one return value." 
                            "The provided function returns" + 
                            std::to_string(specializedFunc.getType().getNumResults()) +
                            "values instead."
                        );

                    // Get current mapOp result matrix type and fix it if needed.
                    // If we fixed something we rerun inference of the whole function
                    daphne::MatrixType resMatrixTy = mapOp.getType().dyn_cast<daphne::MatrixType>();
                    mlir::Type funcResTy = specializedFunc.getType().getResult(0);

                    // The matrix that results from the mapOp has the same dimension as the input 
                    // matrix and the element-type returned by the specialized function
                    if(resMatrixTy.getNumCols() != inpMatrixTy.getNumCols() || 
                        resMatrixTy.getNumRows() != inpMatrixTy.getNumRows() ||
                        resMatrixTy.getElementType() != funcResTy) {
                        mapOp.getResult().setType(inpMatrixTy.withElementType(funcResTy));
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

/**
 * @brief Generate and call specialized functions from template definitions and remove templates.
 *
 * We start entry functions (like `main` or `dist`) and then proceed as follows:
 *
 * 1. Infer types (types up to the first `GenericCallOp` will be inferred for sure)
 * 2. If the function called by `GenericCallOp` is untyped (input types are unknown), we clone it and set the input types
 *      to the types used in the call. For this specialized function we then do the same steps starting at 1.
 * 3. With the (possibly cloned) specialized function we now know the outputs. Starting here we infer up to the next
 *      `GenericCallOp` and go back to step 2.
 * 4. When all `GenericCallOp`s are specialized we are finished
 *
 * Finally we delete all the template functions such that the MLIR code can be verified for correct input and output types.
 */
void SpecializeGenericFunctionsPass::runOnOperation() {
    auto module = getOperation();

    module.walk([&](FuncOp funcOp) {
        functions.insert({funcOp.sym_name().str(), funcOp});
    });

    // `entryFunctions` will hold entry functions like `main`, but also `dist` (for distributed computation)
    // we could also directly specify the names `main`, `dist` etc. (if we add more `entry` functions), or just set
    // an attribute flag for those functions.
    std::vector<FuncOp> entryFunctions;
    for(const auto &entry : functions) {
        entryFunctions.push_back(entry.second);
    }
    for(const auto &function : entryFunctions) {
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
