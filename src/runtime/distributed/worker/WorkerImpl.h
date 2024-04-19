/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
#define SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H

#include <map>

#include <mlir/IR/BuiltinTypes.h>

#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/datastructures/DenseMatrix.h>

class WorkerImpl  
{
public:
    class Status {
        private:
            bool ok_;
            std::string error_message_;
        public:
            Status (bool ok) : ok_(ok), error_message_("") { };
            Status(bool ok, std::string msg) : ok_(ok), error_message_(msg) { };
            bool ok() const { return ok_; };
            std::string error_message() const { return error_message_; }; 
    };
    
    const static std::string DISTRIBUTED_FUNCTION_NAME;

    DaphneUserConfig& cfg;

    WorkerImpl(DaphneUserConfig& _cfg);
    ~WorkerImpl() = default;
    
    virtual void Wait() { };
   
    struct StoredInfo {
        std::string identifier;
        size_t numRows, numCols;
        std::string toString() const {
            return identifier + "," + std::to_string(numRows) + "," + std::to_string(numCols);
        }
    };

    /**
     * @brief Stores a matrix at worker's memory
     * 
     * @param mat Structure * obj to store
     * @return StoredInfo Information regarding stored object (identifier, numRows, numCols)
     */
    template<class DT>
    StoredInfo Store(DT *mat) ;
    
    /**
     * @brief Computes a pipeline
     * 
     * @param outputs vector to populate with results of the pipeline (identifier, numRows/cols, etc.)
     * @param inputs vector with inputs of pipeline (identifiers to use, etc.)
     * @param mlirCode mlir code fragment
     * @return WorkerImpl::Status contains if everything went fine, with an optional error message
     */
    WorkerImpl::Status Compute(std::vector<WorkerImpl::StoredInfo> *outputs,
            const std::vector<WorkerImpl::StoredInfo> &inputs, const std::string &mlirCode);

    /**
     * @brief Returns a matrix stored in worker's memory
     * 
     * @param storedInfo Information regarding stored object (identifier, numRows, numCols)
     * @return Structure* Returns object
     */
    Structure * Transfer(StoredInfo storedInfo);

private:
    uint64_t tmp_file_counter_ = 0;
    std::unordered_map<std::string, void *> localData_;
    /**
     * Creates a vector holding pointers to the inputs as well as the outputs. This vector can directly be passed
     * to the `ExecutionEngine::invokePacked` method.
     * @param functionType Type of the function that will be invoked
     * @param workInputs Inputs send by client
     * @param outputs Reference to the vector that will hold the outputs of the invoked function
     * @return packed pointers to inputs and outputs
     */
    std::vector<void *> createPackedCInterfaceInputsOutputs(mlir::FunctionType functionType,
                                                            std::vector<WorkerImpl::StoredInfo> workInputs,
                                                            std::vector<void *> &outputs,
                                                            std::vector<void *> &inputs);
    
    Structure *readOrGetMatrix(const std::string &identifier, size_t numRows, size_t numCols, bool isSparse = false, bool isFloat = false, bool isScalar = false);
    void *loadWorkInputData(mlir::Type mlirType, StoredInfo& workInput);    
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
