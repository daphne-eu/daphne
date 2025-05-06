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

#include <compiler/utils/CompilerUtils.h>
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <regex>
#include <uuid/uuid.h>
#include <openssl/sha.h>

using namespace mlir;

/**
 * @brief Specialize functions by calculating intermediate results (including return value) inside functions in compile time and 
 * create new specialized functions. It also makes sure to not create too many specializations (deep recursion), maintains a call graph, a hashmap of function bodies
 * and a set of recursive cycles to decide whether it makes sense to specialize or not.
 */

namespace {
    
    /**
     * @brief Generates a SHA-512 hash of the input IR (Intermediate Representation) string.
     *
     * This function computes the SHA-512 hash of a given string representing an
     * Intermediate Representation (IR) to create a unique identifier for the IR.
     * This can be useful for detecting changes in the IR or for comparing two
     * IR functions.
     *
     * @param irString The IR string to hash.
     * @return A hexadecimal string representation of the SHA-512 hash.
     */
    std::string hashIRString(const std::string &irString) {
        unsigned char hash[SHA512_DIGEST_LENGTH];
        SHA512((unsigned char*)irString.c_str(), irString.size(), hash);

        std::stringstream ss;
        for (int i = 0; i < SHA512_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }

    // Function to generate a UUID string
    std::string generateUUID() {
        uuid_t uuid;
        uuid_generate(uuid);
        char uuidStr[37];   // UUIDs are 36 characters + null terminator
        uuid_unparse(uuid, uuidStr);
        return std::string(uuidStr);
    }

    std::string getIRString(func::FuncOp funcOp) {
        std::string irString;
        llvm::raw_string_ostream stream(irString);
        funcOp.print(stream);
        return irString;
    }

    /**
     * @brief Split any string into a vector of lines at all linebreaks of the string
     * 
     * @param str The string to split
     * @return The vector containing all lines
     */
    std::vector<std::string> splitIntoLines(const std::string &str) {
        std::vector<std::string> lines;
        std::stringstream ss(str);
        std::string line;

        while (std::getline(ss, line, '\n')) {
            lines.push_back(line);
        }
        return lines;
    }
    /**
     * @brief Checks if the function is untyped, i.e., if at least one of the inputs is
     * of unknown type.
     * 
     * @param op The `FuncOp` to check
     * @return true if `FuncOp` is untyped, false otherwise
     */
    bool isUntypedFunction(func::FuncOp op) {
        return llvm::any_of(
                op.getFunctionType().getInputs(),
                [&](Type ty) {
                    auto matTy = ty.dyn_cast<daphne::MatrixType>();
                    return
                        llvm::isa<daphne::UnknownType>(ty) ||
                        (matTy && (llvm::isa<daphne::UnknownType>(matTy.getElementType())));
                }
        );
    }

    /**
     * @brief Checks if the function is a template, by checking the types of input arguments.
     * 
     * We consider a function a template iff:
     * (1) it is an untyped function (i.e., at least one of the inputs is of unknown type
     *     or a matrix of unknown value type), or
     * (2) at least one of the inputs is a matrix with unknown properties
     * 
     * @param op The `FuncOp` to check
     * @return true if `FuncOp` is a template, false otherwise
     */
    bool isFunctionTemplate(func::FuncOp op) {
        return llvm::any_of(
                op.getFunctionType().getInputs(),
                [&](Type ty) {
                    auto matTy = ty.dyn_cast<daphne::MatrixType>();
                    return
                        llvm::isa<daphne::UnknownType>(ty) ||
                        (matTy && (
                            llvm::isa<daphne::UnknownType>(matTy.getElementType()) ||
                            (matTy.getNumRows() == -1 && matTy.getNumCols() == -1 && matTy.getSparsity() == -1)
                        ));
                }
        );
    }

    /**
     * @brief Create a unique specialized name for the function. If the specialized function has been created already, the same name gets returned
     *
     * @param functionName The name of the template Function
     * @param inputTypes The input types
     * @param inputValues The input values
     * @return A unique function name for the specialized function 
    */
    std::string uniqueSpecializedFuncName(const std::string &functionName, TypeRange inputTypes, ValueRange inputValues) {
        //static unsigned functionUniqueId = 0;
        // Creating an empty string to store the new unique specialized function name
        std::string name = functionName;

        // Iterating over types and values to use them
        for (auto it : llvm::enumerate(llvm::zip(inputTypes, inputValues))) {
            //auto index = it.index();
            auto value = std::get<1>(it.value());


            // Converting value to string
            std::string valueStr;
            llvm::raw_string_ostream valueStream(valueStr);
            value.print(valueStream);
            std::string valueName = valueStream.str();

            // Append type and value to the general name
            name +=  valueName;
        }
        std::string output = functionName + '(';
        size_t pos = 0;
        bool first = true;

        // Loop through the string and find "value = "
        while ((pos = name.find("value = ", pos)) != std::string::npos) {
            if (!first) {
                output += ",";
            }
            pos += 8; // Move position past "value = "
            size_t end_pos = name.find(" ", pos);
            output += name.substr(pos, end_pos - pos);
            first = false;
        }

        output += ")";
        return output;
    }

    /**
     * @brief Get argument types for the specialized version of a template function.
     * @param functionType The types of the template function.
     * @param callTypes The types used in the call to the specialized version.
     * @param funcName The name of the function to call
     * @param callLoc The location of the call
     * @return The argument types to use for the specialized version
     */
    std::vector<Type> getSpecializedFuncArgTypes(FunctionType functionType, TypeRange callTypes, const std::string & funcName, mlir::Location callLoc) {
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
                bool isMatchingUnknownPropertiesMatrix =
                    funcMatTy && specializedMatTy && funcMatTy.getElementType() == specializedMatTy.getElementType() &&
                    funcMatTy.getNumRows() == -1 && funcMatTy.getNumCols() == -1 && funcMatTy.getSparsity() == -1;
                if(!isMatchingUnknownMatrix && !isMatchingUnknownPropertiesMatrix && funcInTy != unknownTy) {
                    std::string s;
                    llvm::raw_string_ostream stream(s);
                    // TODO The function name funcName has a cryptic suffix from overloading/specialization, which is not suitable for users for see.
                    // TODO This error message can shiw up even for typed functions which are no "templates", which is confusing for a user.
                    // TODO The index seems to be off by 1 (too large)... (or not, simply 0-based counting).
                    stream << "call to function template `" << funcName << "` with invalid types for argument " << index
                           << ": expected `" << funcInTy << "`, got `" << specializedTy << "`";
                    throw ErrorHandler::compilerError(callLoc, "SpecializeGenericFunctionsPass", stream.str());
                }
            }
            // Note that specializedTy may explicitly contain property information (e.g., shape).
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
    func::FuncOp inferTypesInFunction(func::FuncOp function) {
        // Run inference
        mlir::PassManager pm(function->getContext(), "func.func");
        pm.enableVerifier(false);
        // TODO There is a cyclic dependency between (shape) inference and
        // constant folding (included in canonicalization), at the moment we
        // run only three iterations of both passes (see #173).
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        if(failed(pm.run(function))) {
            throw ErrorHandler::compilerError(
                function.getOperation(), "SpecializeGenericFunctionsPass",
                "could not infer types for a call of function template: " +
                    function.getName().str());
        }
        return function;
    }

    class SpecializeGenericFunctionsPass
        : public PassWrapper<SpecializeGenericFunctionsPass, OperationPass<ModuleOp>> {
        std::unordered_map<std::string, func::FuncOp> functions;
        std::multimap<std::string, func::FuncOp> specializedVersions;
        std::set<func::FuncOp> visited;
        std::set<func::FuncOp> called;
        std::set<func::FuncOp> templateFunctions;

        std::map<std::string, std::set<std::string>> callGraph;
        std::set<std::set<std::string>> recursiveCalls;
        std::map<std::set<std::string>, int> recursiveCallsNum;
        std::set<std::string> visitedInGraph;
        std::vector<std::string> callStack;
        std::map<std::string, std::string> duplicateFunctions;
        std::unordered_map<std::string, func::FuncOp> hashToFuncMap;
        std::map<func::FuncOp, std::vector<std::string>> IRRepresentations;
        // Can store all callOps in a list

        // Map to track all functions that share the same base hash (including those with newHash)
        std::unordered_map<std::string, std::set<func::FuncOp>> baseHashMap;
        
        const DaphneUserConfig& userConfig;
        std::shared_ptr<spdlog::logger> logger;

    public:
        explicit SpecializeGenericFunctionsPass(const DaphneUserConfig& cfg) : userConfig(cfg) {
            logger = spdlog::get("compiler");
        }

    private:

        /**
         * @brief Detects recursion in the callStack (Note: This function gets utilized by findRecursions)
         *
         * @param func The function name to find it in the callStack
         * @param visitedInGraph already visited functions
         * @param callStack the callStack
         *
         */
        void detectRecursion(const std::string &func, std::set<std::string> &visitedInGraph, std::vector<std::string> &callStack) {
            // If function is already on the stack, we found a recursion
            auto it = std::find(callStack.begin(), callStack.end(), func);
            if (it != callStack.end()) {
                // Extract the recursion cycle
                std::set<std::string> cycle(it, callStack.end());
                recursiveCalls.insert(cycle);
                return;
            }

            // If function was already visited and didn't form a cycle, skip it
            if (visitedInGraph.find(func) != visitedInGraph.end()) {
                return;
            }

            // Mark the function as visited and add it to the current call stack
            visitedInGraph.insert(func);
            callStack.push_back(func);

            // Recursively visit all called functions
            if (callGraph.find(func) != callGraph.end()) {
                for (const auto &calledFunc : callGraph[func]) {
                    detectRecursion(calledFunc, visitedInGraph, callStack);
                }
            }

            // Backtrack: remove the function from the call stack
            callStack.pop_back();
        }

        /** @brief Find recursion in the whole call graph (using DFS algorithm) and adds them to recursiveCalls
         *  
         */
        void findRecursions() {
            // Clear previous recursive calls
            recursiveCalls.clear();

            // Iterate over each function in the call graph
            for (const auto &entry : callGraph) {
                // Reset visited and call stack for each new starting function
                std::set<std::string> visitedInGraph;
                std::vector<std::string> callStack;
                
                // Detect recursions starting from the current function
                detectRecursion(entry.first, visitedInGraph, callStack);
            }
        }

        /**
         * @brief Update the callGraph map
         *
         * @param func The specialized function
         */
        void updateCallGraph(func::FuncOp func) {
            // Get the module containing this function
            std::string funcName = func.getName().str();
            size_t pos = funcName.find('(');
            if (pos != std::string::npos) {
                funcName =  funcName.substr(0, pos);    
            }
            // Initialize the entry for this function in the call graph if not already present
            if (callGraph.find(funcName) == callGraph.end()) {
                callGraph[funcName] = {};
            } else {
                // If it was initialized already, return immediately. Specialized functions always call the same!
                return;
            }
            func.walk([&](Operation *op) {
                // Check if the operation is a custom function call (e.g., "daphne.generic_call")
                if (op->getName().getStringRef() == "daphne.generic_call") {
                    if (auto calleeAttr = op->getAttrOfType<StringAttr>("callee")) {
                        std::string calleeName = calleeAttr.getValue().str();
                        
                        // Extract the input types (operand types)
                        TypeRange inputTypes = op->getOperandTypes();

                        // Extract the input values (operands)
                        ValueRange inputValues = op->getOperands();

                        // Use the operation's name as the base function name

                        // Generate the specialized function name
                        std::string specializedName = uniqueSpecializedFuncName(calleeName , inputTypes, inputValues);
                        size_t pos = specializedName.find('(');
                        if (pos != std::string::npos) {
                            specializedName =  specializedName.substr(0, pos);    
                        }
                        // Print the specialized function name
                        callGraph[funcName].insert(specializedName);
                    }
                }
            });
        }

        /**
         * @brief Create a specialized version of the template function.
         * @param templateFunction The template function.
         * @param specializedTypes The specialized function arguments
         * @param operands The operands of the call operation
         * @return The specialized function
         */
        func::FuncOp createSpecializedFunction(func::FuncOp templateFunction, TypeRange specializedTypes, ValueRange operands) {
            OpBuilder builder(templateFunction);
            auto specializedFunc = templateFunction.clone();
            builder.insert(specializedFunc);
            auto uniqueFuncName = uniqueSpecializedFuncName(templateFunction.getSymName().str(), specializedTypes, operands);
            specializedFunc.setName(uniqueFuncName);
            functions.insert({uniqueFuncName, specializedFunc});

            // change argument types
            specializedFunc
                .setType(builder.getFunctionType(specializedTypes, specializedFunc.getFunctionType().getResults()));
            for(auto it : llvm::zip(specializedFunc.getArguments(), specializedTypes)) {
                std::get<0>(it).setType(std::get<1>(it));
            }

            bool insertedConst = false;
            // Don't propagate constants into untyped functions, since that still causes problems for some reason.
            if(userConfig.use_ipa_const_propa && !isUntypedFunction(templateFunction)) {
                // Insert compile-time constant scalar call operands into the function.
                Block & specializedFuncBodyBlock = specializedFunc.getBody().front();
                builder.setInsertionPointToStart(&specializedFuncBodyBlock);
                for(auto it : llvm::enumerate(operands)) {
                    auto i = it.index();
                    Value v = it.value();
                    if(Operation * co = CompilerUtils::constantOfAnyType(v)) {
                        // Clone the constant operation into the function body.
                        Operation * coNew = co->clone();
                        builder.insert(coNew);
                        // Replace all uses of the corresponding block argument by the newly inserted constant.
                        specializedFuncBodyBlock.getArgument(i).replaceAllUsesWith(coNew->getResult(0));
                        // TODO We could even remove the corresponding function argument.
                        insertedConst = true;
                    }
                }
            }
            // Remember the newly specialized function for reuse only if we did not insert any constant
            // call operands.
            // TODO We could reuse it for other calls with the same constant (it's just more book-keeping effort).
            if(!insertedConst)
                specializedVersions.insert({templateFunction.getSymName().str(), specializedFunc});
            
            updateCallGraph(inferTypesInFunction(specializedFunc));
            findRecursions();
            IRRepresentations[specializedFunc] = splitIntoLines(getIRString(specializedFunc));
            return inferTypesInFunction(specializedFunc);
        }

        /**
         * @brief Try to reuse an existing specialization for the given template function
         * @param specializedName The specialized name following the convention
         * @return either an existing and matching `FuncOp`, `nullptr` otherwise
         */
        func::FuncOp tryReuseExistingSpecialization(std::string specializedName) {
            auto it = functions.find(specializedName);
            if(it != functions.end()) {
                return it->second;
            }
            return nullptr;
        }

        /**
         * @brief Try to reuse an existing specializtion if one exists, else creates a new 
         *  specialization
         * @param operandTypes Operand types of the call operation
         * @param operands Operands of the call operation or an empty list if the operands are not available
         * @param calledFunction The function called by the call operation
         * @param callLoc The location of the call for which a function specialization shall be created or reused
         * @return A `FuncOp`for the specialization
         */
        func::FuncOp createOrReuseSpecialization(TypeRange operandTypes, ValueRange operands, func::FuncOp calledFunction, mlir::Location callLoc) {
            // check for existing specialization that matches

            auto specializedTypes = getSpecializedFuncArgTypes(calledFunction.getFunctionType(), operandTypes, calledFunction.getSymName().str(), callLoc); 
            auto specializedName = uniqueSpecializedFuncName(calledFunction.getSymName().str(), specializedTypes,operands );
            func::FuncOp specializedFunc = tryReuseExistingSpecialization(specializedName);
            if(!specializedFunc) {
                // Create specialized function
                specializedFunc = createSpecializedFunction(calledFunction, specializedTypes, operands);
            }
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                calledFunction->getLoc().print(stream);
                logger->debug("calledFunction\n\tname: {}\n\tlocation: {}", calledFunction.getSymName().str(), s);
            }
            templateFunctions.insert(calledFunction);
            return specializedFunc;
        }

        /**
         * @brief Recursively specializes all functions within a `FuncOp` based on calls to the functions. And also ...
         * ... keep track of how often recursive calls had happened and prevent recursion if too many recursive calls happened in a cycle.
         * ... keep track of a function hashmap which is used to find and delete duplicate functions with different function names (prevent non beneficial specialization)
         * @param function The `FuncOp` to scan for function specializations
         */
        void specializeCallsInFunction(func::FuncOp function) {

            if(visited.count(function)) {
                return;
            }
            visited.insert(function);

            // Specialize all functions called directly
            function.walk([&](daphne::GenericCallOp callOp) {
                auto calledFunction = functions[callOp.getCallee().str()];
                bool hasConstantInput = llvm::any_of(
                        callOp.getOperands(),
                        [&](Value v) {
                            return CompilerUtils::constantOfAnyType(v) != nullptr;
                        }
                );

                if(isFunctionTemplate(calledFunction) || hasConstantInput) {
                    auto specializedTypes = getSpecializedFuncArgTypes(calledFunction.getFunctionType(), callOp.getOperandTypes(), calledFunction.getSymName().str(), callOp.getLoc()); 
                    std::string calledFuncName;
                    std::string specializedName = uniqueSpecializedFuncName(calledFunction.getSymName().str(), specializedTypes,callOp.getOperands());
                    std::set<std::string> callPath;
                    std::string functionName = function.getName().str();


                    // Begin section checking for recursion

                    // Clean the func names from the specialized part (we only look at template functions!)
                    size_t pos = specializedName.find('('); // we just want a-1 and not a-1(1)
                    if (pos != std::string::npos) {
                        calledFuncName =  specializedName.substr(0, pos);    
                    }
                    pos = functionName.find('(');
                    if (pos != std::string::npos) {
                        functionName =  functionName.substr(0, pos);    
                    }

                    callPath.insert(calledFuncName);
                    callPath.insert(functionName);
                    bool specialize=true;
                    for (auto it : recursiveCalls) {
  
                        // Check if `callPath` is a subset of `it`
                        if (std::includes(it.begin(), it.end(), callPath.begin(), callPath.end())) {
                            if(recursiveCallsNum[it] >= 5) {
                                specialize=false;
                            }
                            recursiveCallsNum[it] += 1;
                        } 
                    }


                    // Begin section duplicate function detection


                    if (specialize) {
                        // Canonicalize and add template function to baseHashMap (because specialization could be non beneficial)
                        mlir::PassManager pm(function.getContext());
                        pm.addPass(mlir::createCanonicalizerPass());
                        if (failed(pm.run(calledFunction))) {
                            throw std::runtime_error("Failed to canonicalize called function.");
                        }
                        std::vector<std::string> templateFunctionLines = splitIntoLines(getIRString(calledFunction));
                        size_t firstQuote = templateFunctionLines[0].find('"');
                        if(firstQuote != std::string::npos) {
                            size_t secondQuote = templateFunctionLines[0].find('"', firstQuote+1);
                            if(secondQuote != std::string::npos) {
                                templateFunctionLines[0] = templateFunctionLines[0].substr(secondQuote+1);
                            }
                        }
                        std::string templateFunctionIR = std::accumulate(templateFunctionLines.begin(), templateFunctionLines.end(), std::string());
                        std::string templateFuncHash = hashIRString(templateFunctionIR);
                        baseHashMap[templateFuncHash].insert(calledFunction);
                        IRRepresentations[calledFunction] = templateFunctionLines;
                        // Canonicalize the called function using MLIR's built-in canonicalizer
                        func::FuncOp specializedFunc = createOrReuseSpecialization(callOp.getOperandTypes(), callOp.getOperands(), calledFunction, callOp.getLoc());

                        // Get the canonicalized IR as a string
                        std::vector<std::string> canonicalIRLines = splitIntoLines(getIRString(specializedFunc));

                        // Only keep function signature, ignore name 
                        firstQuote = canonicalIRLines[0].find('"');
                        if(firstQuote != std::string::npos) {
                            size_t secondQuote = canonicalIRLines[0].find('"', firstQuote+1);
                            if(secondQuote != std::string::npos) {
                                canonicalIRLines[0] = canonicalIRLines[0].substr(secondQuote+1);
                            }
                        }

                        std::string canonicalIR = std::accumulate(canonicalIRLines.begin(), canonicalIRLines.end(), std::string());

                        std::string baseHash = hashIRString(canonicalIR);
                        std::string finalHash = baseHash;
                        bool isDuplicate = false;

                        // Check if the base hash already exists in baseHashMap
                        if (baseHashMap.find(baseHash) != baseHashMap.end()) {
                            // Iterate through functions that share the same base hash
                            for (func::FuncOp existingFunc : baseHashMap[baseHash]) {
                                std::vector<std::string> existingIRLines = IRRepresentations[existingFunc];
                                std::string existingCanonicalIR = std::accumulate(existingIRLines.begin(), existingIRLines.end(), std::string());

                                if (canonicalIR == existingCanonicalIR && existingFunc.getName().str() != specializedFunc.getName().str()) {
                                    // Functions are identical, reuse the existing one
                                    callOp.setCalleeAttr(existingFunc.getSymNameAttr());
                                    if (fixResultTypes(callOp->getResults(), existingFunc.getFunctionType())) {
                                        inferTypesInFunction(function);
                                    }
                                    specializedFunc.erase();
                                    functions.erase(specializedName);
                                    called.insert(existingFunc);
                                    isDuplicate = true;
                                    break;
                                }
                            }

                            // If no identical function found, generate a UUID for a new unique hash
                            if (!isDuplicate) {
                                std::string uuid = generateUUID(); // Use UUID for uniqueness
                                finalHash = baseHash + "_" + uuid;
                            }
                        } else {
                            // If the baseHash does not exist in baseHashMap, add the function directly
                            finalHash = baseHash;  // No need for UUID as there's no collision
                        }

                        // If no duplicate found, store the new function with its unique hash
                        if (!isDuplicate) {
                            callOp.setCalleeAttr(specializedFunc.getSymNameAttr());
                            if (fixResultTypes(callOp->getResults(), specializedFunc.getFunctionType())) {
                                inferTypesInFunction(function);
                            }
                            specializeCallsInFunction(specializedFunc);
                            called.insert(specializedFunc);

                            // Store the function in both hashToFuncMap and baseHashMap
                            hashToFuncMap[finalHash] = specializedFunc;
                            baseHashMap[baseHash].insert(specializedFunc);
                            IRRepresentations[specializedFunc] = canonicalIRLines;
                        }
                    } else {
                        functions.insert({calledFuncName, calledFunction});

                        // Keep track of the hash from template function because specialization could be non beneficial

                        callOp.setCalleeAttr(calledFunction.getSymNameAttr());
                        if (fixResultTypes(callOp->getResults(), calledFunction.getFunctionType())) {
                            inferTypesInFunction(function);
                        }
                        specializeCallsInFunction(calledFunction);
                        called.insert(calledFunction);
                    }
                }
                else {
                    specializeCallsInFunction(calledFunction);
                    called.insert(calledFunction);
                }
            });

            // Specialize all functions called by MapOp
            function.walk([&](daphne::MapOp mapOp) {
                auto calledFunction = functions[mapOp.getFunc().str()];
                if(isFunctionTemplate(calledFunction)) {
                     // Get the element type of the matrix the function should be mapped on
                    mlir::Type opTy = mapOp.getArg().getType();
                    auto inpMatrixTy = opTy.dyn_cast<daphne::MatrixType>();
                    func::FuncOp specializedFunc = createOrReuseSpecialization(inpMatrixTy.getElementType(), {}, calledFunction, mapOp.getLoc());
                    mapOp.setFuncAttr(specializedFunc.getSymNameAttr());

                    // We only allow functions that return exactly one result for mapOp
                    if (specializedFunc.getFunctionType().getNumResults() != 1) {
                        throw ErrorHandler::compilerError(
                            mapOp.getOperation(),
                            "SpecializeGenericFunctionsPass",
                            "map expects a function with exactly one return "
                            "value. The provided function returns" +
                                std::to_string(specializedFunc.getFunctionType()
                                                   .getNumResults()) +
                                "values instead.");
                    }

                    // Get current mapOp result matrix type and fix it if needed.
                    // If we fixed something we rerun inference of the whole function
                    daphne::MatrixType resMatrixTy = mapOp.getType().dyn_cast<daphne::MatrixType>();
                    mlir::Type funcResTy = specializedFunc.getFunctionType().getResult(0);

                    // The matrix that results from the mapOp has the same dimension as the input 
                    // matrix and the element-type returned by the specialized function
                    if(resMatrixTy.getNumCols() != inpMatrixTy.getNumCols() || 
                        resMatrixTy.getNumRows() != inpMatrixTy.getNumRows() ||
                        resMatrixTy.getElementType() != funcResTy) {
                        mapOp.getResult().setType(inpMatrixTy.withElementType(funcResTy));
                        inferTypesInFunction(function);
                    }

                    specializeCallsInFunction(specializedFunc);
                    called.insert(specializedFunc);
                }
                else {
                    specializeCallsInFunction(calledFunction);
                    called.insert(calledFunction);
                }
            });
        }

    public:
        void runOnOperation() final;

    StringRef getArgument() const final { return "specialize-generic-funcs"; }
    StringRef getDescription() const final { return "TODO"; }
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

    module.walk([&](func::FuncOp funcOp) {
        functions.insert({funcOp.getSymName().str(), funcOp});
    });

    // `entryFunctions` will hold entry functions like `main`, but also `dist` (for distributed computation)
    // we could also directly specify the names `main`, `dist` etc. (if we add more `entry` functions), or just set
    // an attribute flag for those functions.
    std::vector<func::FuncOp> entryFunctions;
    for(const auto &entry : functions) {
        entryFunctions.push_back(entry.second);
    }

    for(const auto &function : entryFunctions) {
        if(isFunctionTemplate(function) || visited.count(function) || templateFunctions.count(function))
            continue;
        try {
            inferTypesInFunction(function);
        } catch (std::runtime_error& e) {
            throw ErrorHandler::rethrowError("SpecializeGenericFunctionsPass", e.what());
        }
        specializeCallsInFunction(function);
    }

    // Delete non-called functions.
    for(auto f : functions) {
        // Never remove the main or dist function.
        if(f.first == "main" or f.first == "dist")
            continue;
        // Remove a function that was present before creating specializations,
        // if it is never called.
        if(!called.count(f.second)) {
            f.second.erase();
        }
    }
}

std::unique_ptr<Pass> daphne::createSpecializeGenericFunctionsPass(const DaphneUserConfig& cfg) {
    
    return std::make_unique<SpecializeGenericFunctionsPass>(cfg);
}
