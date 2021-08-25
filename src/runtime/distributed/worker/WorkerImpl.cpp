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

#include <runtime/distributed/worker/ProtoDataConverter.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/File.h>
#include <compiler/execution/DaphneIrExecutor.h>

const std::string WorkerImpl::DISTRIBUTED_FUNCTION_NAME = "dist";

WorkerImpl::WorkerImpl() : tmp_file_counter_(0), localData_()
{
}

WorkerImpl::~WorkerImpl() = default;

grpc::Status WorkerImpl::Store(::grpc::ServerContext *context,
                               const ::distributed::Matrix *request,
                               ::distributed::StoredData *response)
{
    auto *mat = DataObjectFactory::create<DenseMatrix<double>>(request->num_rows(), request->num_cols(), false);
    ProtoDataConverter::convertFromProto(*request, mat);

    auto identification = "tmp_" + std::to_string(tmp_file_counter_++);
    localData_[identification] = mat;

    response->set_filename(identification);
    response->set_num_rows(mat->getNumRows());
    response->set_num_cols(mat->getNumCols());
    return ::grpc::Status::OK;
}

grpc::Status WorkerImpl::Compute(::grpc::ServerContext *context,
                                 const ::distributed::Task *request,
                                 ::distributed::ComputeResult *response)
{
    // TODO Decide if vectorized pipelines should be used on this worker.
    DaphneIrExecutor executor(false, false);

    mlir::OwningModuleRef module(mlir::parseSourceString<mlir::ModuleOp>(request->mlir_code(), executor.getContext()));
    if (!module) {
        auto message = "Failed parse source string.\n";
        llvm::errs() << message;
        return ::grpc::Status(::grpc::StatusCode::ABORTED, message);
    }

    auto *distOp = module->lookupSymbol(DISTRIBUTED_FUNCTION_NAME);
    mlir::FuncOp distFunc;
    if (!(distFunc = llvm::dyn_cast_or_null<mlir::FuncOp>(distOp))) {
        auto message = "MLIR fragment has to contain `dist` FuncOp\n";
        llvm::errs() << message;
        return ::grpc::Status(::grpc::StatusCode::ABORTED, message);
    }
    auto distFuncTy = distFunc.getType();

    std::vector<void *> inputs;
    std::vector<void *> outputs;
    auto packedInputsOutputs = createPackedCInterfaceInputsOutputs(distFuncTy,
        request->inputs(),
        outputs,
        inputs);

    // Execution
    if (!executor.runPasses(module.get())) {
        std::stringstream ss;
        ss << "Module Pass Error.\n";
        // module->print(ss, llvm::None);
        llvm::errs() << ss.str();
        return ::grpc::Status(::grpc::StatusCode::ABORTED, ss.str());
    }

    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto engine = executor.createExecutionEngine(module.get());
    if (!engine) {
        return ::grpc::Status(::grpc::StatusCode::ABORTED, "Failed to create JIT-Execution engine");
    }
    auto error = engine->invokePacked(DISTRIBUTED_FUNCTION_NAME,
        llvm::MutableArrayRef<void *>{&packedInputsOutputs[0], (size_t)0});

    if (error) {
        std::stringstream ss("JIT-Engine invocation failed.");
        llvm::errs() << "JIT-Engine invocation failed: " << error << '\n';
        return ::grpc::Status(::grpc::StatusCode::ABORTED, ss.str());
    }

    for (auto zipped : llvm::zip(outputs, distFuncTy.getResults())) {
        auto output = std::get<0>(zipped);
        auto type = std::get<1>(zipped);

        auto identification = "tmp_" + std::to_string(tmp_file_counter_++);
        localData_[identification] = output;

        distributed::WorkData::DataCase dataCase = dataCaseForType(type);

        distributed::WorkData workData;
        switch (dataCase) {
        case distributed::WorkData::kStored: {
            auto mat = static_cast<DenseMatrix<double> *>(output);

            workData.mutable_stored()->set_filename(identification);
            workData.mutable_stored()->set_num_rows(mat->getNumRows());
            workData.mutable_stored()->set_num_cols(mat->getNumCols());
            break;
        }
        default: assert(false);
        }
        *response->add_outputs() = workData;
    }
    // TODO: cache management (Write to file/evict matrices present as files)
    return ::grpc::Status::OK;
}

distributed::WorkData::DataCase WorkerImpl::dataCaseForType(mlir::Type type)
{
    distributed::WorkData::DataCase dataCase;
    if (type.isa<mlir::daphne::MatrixType>()) {
        dataCase = distributed::WorkData::kStored;
    }
    else {
        // TODO: further types data cases
        assert(false && "TODO");
    }
    return dataCase;
}

grpc::Status WorkerImpl::Transfer(::grpc::ServerContext *context,
                                  const ::distributed::StoredData *request,
                                  ::distributed::Matrix *response)
{
    auto *mat = readOrGetMatrix(request->filename(), request->num_rows(), request->num_cols());
    ProtoDataConverter::convertToProto(mat, response);
    return ::grpc::Status::OK;
}

std::vector<void *> WorkerImpl::createPackedCInterfaceInputsOutputs(mlir::FunctionType functionType,
                                                                    google::protobuf::RepeatedPtrField<distributed::WorkData> workInputs,
                                                                    std::vector<void *> &outputs,
                                                                    std::vector<void *> &inputs)
{
    assert(functionType.getNumInputs() == workInputs.size()
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

    for (const auto &type : functionType.getResults()) {
        outputs.push_back(nullptr);
        inputsAndOutputs.push_back(&outputs.back());
    }
    return inputsAndOutputs;
}

void *WorkerImpl::loadWorkInputData(mlir::Type mlirType, const distributed::WorkData &workInput)
{
    switch (workInput.data_case()) {
    case distributed::WorkData::kStored: {
        const auto &stored = workInput.stored();
        // TODO: all types
        auto matTy = mlirType.dyn_cast<mlir::daphne::MatrixType>();
        assert(matTy && matTy.getElementType().isa<mlir::Float64Type>() && "We only support double matrices for now!");
        return readOrGetMatrix(stored.filename(), stored.num_rows(), stored.num_cols());
    }
    default:assert(false && "We only support stored data for now");
    }
}

DenseMatrix<double> *WorkerImpl::readOrGetMatrix(const std::string &filename, size_t numRows, size_t numCols)
{
    DenseMatrix<double> *m = nullptr;
    auto data_it = localData_.find(filename);
    if (data_it != localData_.end()) {
        // Data already cached
        m = static_cast<DenseMatrix<double> *>(data_it->second);
    }
    else {
        // Data not yet loaded -> load from file
        struct File *file = openFile(filename.c_str());
        char delim = ',';
        readCsv(m, file, numRows, numCols, delim);
        auto result = localData_.insert({filename, m});
        assert(result.second && "Value should always be inserted");
    }
    return m;
}

