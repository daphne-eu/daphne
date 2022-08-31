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

#include <runtime/local/datastructures/DenseMatrix.h>

class WorkerImpl  
{
public:
    const static std::string DISTRIBUTED_FUNCTION_NAME;
   
    WorkerImpl();
    ~WorkerImpl();
    
    virtual void Wait() = 0;
    struct StoredInfo {
        std::string filename;
        size_t numRows, numCols;
    };

    /**
     * @brief Stores a matrix at worker's memory
     * 
     * @param mat Structure * obj to store
     * @return StoredInfo Information regarding stored object (filename, numRows, numCols)
     */
    StoredInfo Store(Structure *mat) ;
    
    /**
     * @brief Computes a pipeline
     * 
     * @param outputs vector to populate with results of the pipeline (filename, numRows/cols, etc.)
     * @param inputs vector with inputs of pipeline (filenames to use, etc.)
     * @param mlirCode mlir code fragment
     * @return std::string Returns "OK" if everything went fine, otherwise returns an abort message
     */
    std::string Compute(std::vector<WorkerImpl::StoredInfo> *outputs, std::vector<WorkerImpl::StoredInfo> inputs, std::string mlirCode) ;

    /**
     * @brief Returns a matrix stored in worker's memory
     * 
     * @param storedInfo Information regarding stored object (filename, numRows, numCols)
     * @return Structure* Returns object
     */
    Structure * Transfer(StoredInfo storedInfo);

    // grpc::Status FreeMem(::grpc::ServerContext *context,
    //                      const ::distributed::StoredData *request,
    //                      ::distributed::Empty *emptyMessage);
    
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
    
    Structure *readOrGetMatrix(const std::string &filename, size_t numRows, size_t numCols, bool isSparse = false, bool isFloat = false);
    void *loadWorkInputData(mlir::Type mlirType, StoredInfo& workInput);
    

    /**
     * @brief Helper FreeMem function using templates in order to handle
     * different types.
     * 
     * @tparam VT double/int etc.
     */
    // TODO this is still tied to grpc must be updated
    // template<typename VT>
    // grpc::Status FreeMemType(::grpc::ServerContext *context,
    //                      const ::distributed::StoredData *request,
    //                      ::distributed::Empty *emptyMessage);
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
