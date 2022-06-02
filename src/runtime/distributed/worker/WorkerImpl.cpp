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

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <llvm/Support/SourceMgr.h>

#include <ir/daphneir/Daphne.h>

#include "WorkerImpl.h"

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/Read.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/File.h>
#include <compiler/execution/DaphneIrExecutor.h>

const std::string WorkerImpl::DISTRIBUTED_FUNCTION_NAME = "dist";

WorkerImpl::WorkerImpl() : tmp_file_counter_(0), localData_()
{
}

WorkerImpl::~WorkerImpl() = default;

// void WorkerImpl::StartHandleThread() {
//     HandleRpcsThread = std::thread(&WorkerImpl::HandleRpcs, this);
// }
// void WorkerImpl::TerminateHandleThread() {
//     cq_->Shutdown();
//     HandleRpcsThread.join();
// }




WorkerImpl::StoredInfo WorkerImpl::Store(Structure *mat)
{    
    auto identification = "tmp_" + std::to_string(tmp_file_counter_++);
    localData_[identification] = mat;
    
    return StoredInfo({identification, mat->getNumRows(), mat->getNumCols()});
}



std::string WorkerImpl::Compute(std::vector<WorkerImpl::StoredInfo> *outputs, std::vector<WorkerImpl::StoredInfo> inputs, std::string mlirCode)
{
    // ToDo: user config
    DaphneUserConfig cfg;
    cfg.use_vectorized_exec = true;
    // TODO Decide if vectorized pipelines should be used on this worker.
    // TODO Decide if selectMatrixReprs should be used on this worker.
    // TODO Once we hand over longer pipelines to the workers, we might not
    // want to hardcode insertFreeOp to false anymore. But maybe we will insert
    // the FreeOps at the coordinator already.
    DaphneIrExecutor executor(false, false, cfg);

    mlir::OwningModuleRef module(mlir::parseSourceString<mlir::ModuleOp>(mlirCode, executor.getContext()));
    if (!module) {
        auto message = "Failed to parse source string.\n";
        llvm::errs() << message;
        return (message);
    }

    auto *distOp = module->lookupSymbol(DISTRIBUTED_FUNCTION_NAME);
    mlir::FuncOp distFunc;
    if (!(distFunc = llvm::dyn_cast_or_null<mlir::FuncOp>(distOp))) {
        auto message = "MLIR fragment has to contain `dist` FuncOp\n";
        llvm::errs() << message;
        return message;
    }
    auto distFuncTy = distFunc.getType();

    std::vector<void *> inputsObj;
    std::vector<void *> outputsObj;
    auto packedInputsOutputs = createPackedCInterfaceInputsOutputs(distFuncTy,
        inputs,
        outputsObj,
        inputsObj);
    
    // Increase the reference counters of all inputs to the `dist` function.
    // (But only consider data objects, not scalars.)
    // This is necessary to avoid them from being destroyed within the
    // function. Note that this increasing is in-line with the treatment of
    // local function calls, where we also increase the inputs' reference
    // counters before the call, for the same reason. See ManageObjsRefsPass
    // for details.
    for(size_t i = 0; i < inputsObj.size(); i++)
        // TODO Use CompilerUtils::isObjType() once this branch has been rebased.
        // if(CompilerUtils::isObjType(distFuncTy.getInput(i)))
        if(distFuncTy.getInput(i).isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>())
            reinterpret_cast<Structure*>(inputsObj[i])->increaseRefCounter();

    // Execution
    // TODO Before we run the passes, we should insert information on shape
    // (and potentially other properties) into the types of the arguments of
    // the DISTRIBUTED_FUNCTION_NAME function. At least the shape can be
    // obtained from the cached data partitions in localData_. Then, shape
    // inference etc. should work within this function.
    if (!executor.runPasses(module.get())) {
        std::stringstream ss;
        ss << "Module Pass Error.\n";
        // module->print(ss, llvm::None);
        llvm::errs() << ss.str();
        return ss.str();
    }

    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto engine = executor.createExecutionEngine(module.get());
    if (!engine) {
        return std::string("Failed to create JIT-Execution engine");
    }
    auto error = engine->invokePacked(DISTRIBUTED_FUNCTION_NAME,
        llvm::MutableArrayRef<void *>{&packedInputsOutputs[0], (size_t)0});

    if (error) {
        std::stringstream ss("JIT-Engine invocation failed.");
        llvm::errs() << "JIT-Engine invocation failed: " << error << '\n';
        return ss.str();
    }

    for (auto zipped : llvm::zip(outputsObj, distFuncTy.getResults())) {
        auto output = std::get<0>(zipped);
        auto type = std::get<1>(zipped);

        auto identification = "tmp_" + std::to_string(tmp_file_counter_++);
        localData_[identification] = output;

        auto mat = static_cast<Structure*>(output);
                    
        outputs->push_back(StoredInfo({identification, mat->getNumRows(), mat->getNumCols()}));
    }
    // TODO: cache management (Write to file/evict matrices present as files)
    return "OK";
}

// distributed::WorkData::DataCase WorkerImpl::dataCaseForType(mlir::Type type)
// {
//     distributed::WorkData::DataCase dataCase;
//     if (type.isa<mlir::daphne::MatrixType>()) {
//         dataCase = distributed::WorkData::kStored;
//     }
//     else {
//         // TODO: further types data cases
//         assert(false && "TODO");
//     }
//     return dataCase;
// }


Structure * WorkerImpl::Transfer(StoredInfo info)
{
    Structure *mat = readOrGetMatrix(info.filename, info.numRows, info.numCols);
    return mat;
}


std::vector<void *> WorkerImpl::createPackedCInterfaceInputsOutputs(mlir::FunctionType functionType,
                                                                    std::vector<WorkerImpl::StoredInfo> workInputs,
                                                                    std::vector<void *> &outputs,
                                                                    std::vector<void *> &inputs)
{
    assert(static_cast<int>(functionType.getNumInputs()) == workInputs.size()
        && "Number of inputs received have to match number of MLIR fragment inputs");
    std::vector<void *> inputsAndOutputs;

    // No realloc is allowed to happen, otherwise the pointers are invalid
    inputs.reserve(workInputs.size());
    outputs.reserve(functionType.getNumResults());

    for (const auto &typeAndWorkInput : llvm::zip(functionType.getInputs(), workInputs)) {
        auto type = std::get<0>(typeAndWorkInput);
        auto workInput = std::get<1>(typeAndWorkInput);

        inputs.push_back(loadWorkInputData(type, workInput));
        inputsAndOutputs.push_back(&inputs.back());
    }

//    for (const auto &type : functionType.getResults()) {
    for(auto i = 0ul; i < functionType.getResults().size(); ++i) {
        outputs.push_back(nullptr);
        inputsAndOutputs.push_back(&outputs.back());
    }
    return inputsAndOutputs;
}

void *WorkerImpl::loadWorkInputData(mlir::Type mlirType, StoredInfo &workInput)
{    
    // TODO: all types
    auto matTy = mlirType.dyn_cast<mlir::daphne::MatrixType>();        
    bool isSparse = matTy.getRepresentation() == mlir::daphne::MatrixRepresentation::Sparse;       
    bool isFloat = matTy.getElementType().isa<mlir::Float64Type>();
    
    return readOrGetMatrix(workInput.filename, workInput.numRows, workInput.numCols, isSparse, isFloat);
}

Structure *WorkerImpl::readOrGetMatrix(const std::string &filename, size_t numRows, size_t numCols, bool isSparse /*= false */, bool isFloat /* = false*/)
{
    auto data_it = localData_.find(filename);
    if (data_it != localData_.end()) {
        // Data already cached
        return static_cast<Structure *>(data_it->second);
    }
    else {
        // Data not yet loaded -> load from file
        Structure * m = nullptr;
        // TODO do we need to check for sparsity here? Why Dense and CSR use different read method?
        if(isSparse) {        
            if (isFloat){
                CSRMatrix<double> *m2 = nullptr;
                read<CSRMatrix<double>>(m2, filename.c_str(), nullptr);
                m = m2;
            }
            else{
                CSRMatrix<int64_t> *m2 = nullptr;
                read<CSRMatrix<int64_t>>(m2, filename.c_str(), nullptr);
                m = m2;
            }
        }
        else {
            struct File *file = openFile(filename.c_str());
            char delim = ',';
            // TODO use read
            if (isFloat) {
                DenseMatrix<double> *m2 = nullptr;                
                readCsvFile<DenseMatrix<double>>(m2, file, numRows, numCols, delim);
                m = m2;
            } else {
                DenseMatrix<int64_t> *m2 = nullptr;                
                readCsvFile<DenseMatrix<int64_t>>(m2, file, numRows, numCols, delim);
                m = m2;
            }
            closeFile(file);
        }
//        auto result = localData_.insert({filename, m});
//        assert(result.second && "Value should always be inserted");
        assert(localData_.insert({filename, m}).second && "Value should always be inserted");
        return m;    
    }
}

// grpc::Status WorkerImpl::FreeMem(::grpc::ServerContext *context,
//                                const ::distributed::StoredData *request,
//                                ::distributed::Empty *emptyMessg)
// {
//     // switch (request->type())
//     // {
//     // case distributed::StoredData::Type::StoredData_Type_DenseMatrix_i64:
//     // case distributed::StoredData::Type::StoredData_Type_CSRMatrix_i64:
//     //     return FreeMemType<int64_t>(context, request, emptyMessg);
//     //     break;
//     // case distributed::StoredData::Type::StoredData_Type_DenseMatrix_f64:
//     // case distributed::StoredData::Type::StoredData_Type_CSRMatrix_f64:
//     //     return FreeMemType<double>(context, request, emptyMessg);
//     //     break;
//     // default:
//     //     break;
//     // }
//     // return grpc::Status::CANCELLED;
// }

// template<typename VT>
// grpc::Status WorkerImpl::FreeMemType(::grpc::ServerContext *context,
//                                const ::distributed::StoredData *request,
//                                ::distributed::Empty *emptyMessg)
// {
//     auto filename = request->filename();
//     auto data_it = localData_.find(filename);

//     if (data_it != localData_.end()) {
//         auto * mat = reinterpret_cast<Matrix<VT> *>(data_it->second);
//         if(auto m = dynamic_cast<DenseMatrix<VT> *>(mat))
//             DataObjectFactory::destroy(m);
//         else if(auto m = dynamic_cast<CSRMatrix<VT> *>(mat))
//             DataObjectFactory::destroy(m);
//         localData_.erase(filename);
//     }
//     return grpc::Status::OK;
// }
