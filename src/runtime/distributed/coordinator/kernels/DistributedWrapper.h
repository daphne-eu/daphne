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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H

#include <ir/daphneir/Daphne.h>

#include <runtime/distributed/coordinator/kernels/Broadcast.h>
#include <runtime/distributed/coordinator/kernels/Distribute.h>
#include <runtime/distributed/coordinator/kernels/DistributedCollect.h>
#include <runtime/distributed/coordinator/kernels/DistributedCompute.h>

#include <runtime/distributed/coordinator/kernels/IAllocationDescriptorDistributed.h>
#include <runtime/distributed/coordinator/kernels/AllocationDescriptorDistributedGRPC.h>



#include <mlir/InitAllDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>
#include <vector>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

template <class DT>
class DistributedWrapper {
private:
    DCTX(_ctx);

protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, const Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }
public:
    DistributedWrapper(DCTX(ctx)) : _ctx(ctx) {
        //TODO start workers from here instead of manually (e.g. resource manager) ? 

    }
    ~DistributedWrapper() = default; //TODO Terminate workers (e.g. with gRPC, resource manager, etc.)

    void execute(const char *mlirCode,
                 DT ***res,
                 const Structure **inputs,
                 size_t numInputs,
                 size_t numOutputs,
                 int64_t *outRows,
                 int64_t *outCols,
                 VectorSplit *splits,
                 VectorCombine *combines)                 
    {        
        auto envVar = std::getenv("DISTRIBUTED_WORKERS");
        // assert(envVar && "Environment variable has to be set");
        std::string workersStr(envVar);        
        std::string delimiter(",");

        size_t pos;
        std::vector<std::string> workers;
        while ((pos = workersStr.find(delimiter)) != std::string::npos) {
            workers.push_back(workersStr.substr(0, pos));
            workersStr.erase(0, pos + delimiter.size());
        }
        workers.push_back(workersStr);
        
        // Backend Implementation 
        // gRPC hard-coded selection
        // TODO choose implementation based on configFile/command-line argument        
        auto alloc_type = ALLOCATION_TYPE::DIST_GRPC;

        // output allocation for row-wise combine
        for(size_t i = 0; i < numOutputs; ++i) {
            if(*(res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                // TODO we know result is only DenseMatrix<double> for now,
                // but in the future this will change to support other DataTypes
                *(res[i]) = DataObjectFactory::create<DT>(outRows[i], outCols[i], zeroOut);
            }
        }
        std::vector<const Structure *> vec;
        for (size_t i = 0; i < numInputs; i++)
            vec.push_back(inputs[i]);
        sort(vec.begin(), vec.end());
        const bool hasDuplicates = std::adjacent_find(vec.begin(), vec.end()) != vec.end();
        if(hasDuplicates)
            throw std::runtime_error("Distributed runtime only supports unique inputs for now (no duplicate inputs in a pipeline)");
        // Parse mlir code fragment to determin pipeline inputs/outputs
        auto inputTypes = getPipelineInputTypes(mlirCode);
        // Distribute and broadcast inputs        
        // Each primitive sends information to workers and changes the Structures' metadata information 
        for (auto i = 0u; i < numInputs; ++i) {
            // if already placed on workers, skip
            // TODO maybe this is not enough. We might also need to check if data resides in the specific way we need to.
            // (i.e. rows/cols splitted accordingly). If it does then we can skip.
            if (isBroadcast(splits[i], inputs[i])){
                auto type = inputTypes.at(i);
                if (type==INPUT_TYPE::Matrix) {
                    broadcast(inputs[i], false, alloc_type, _ctx);
                }
                else {
                    broadcast(inputs[i], true, alloc_type, _ctx);
                }
            }
            else {
                assert(splits[i] == VectorSplit::ROWS && "only row split supported for now");
                // std::cout << i << " distr: " << inputs[i]->getNumRows() << " x " << inputs[i]->getNumCols() << std::endl;
                distribute(inputs[i], alloc_type, _ctx);        
            }
        }

          
        distributedCompute(res, numOutputs, inputs, numInputs, mlirCode, combines, alloc_type, _ctx);

        // Collect
        for (size_t o = 0; o < numOutputs; o++){
            assert ((combines[o] == VectorCombine::ROWS || combines[o] == VectorCombine::COLS) && "we only support rows/cols combine atm");
            distributedCollect(*res[o], alloc_type, _ctx);           
        }
        
        
    }

private:
    enum INPUT_TYPE {
        Matrix,
        Double,
        // TOOD add more
    };
    std::vector<INPUT_TYPE> getPipelineInputTypes(const char *mlirCode)
    {
        // is it safe to pass null for mlir::DaphneContext? 
        // Fixme: is it ok to allow unregistered dialects?
        mlir::MLIRContext ctx;
        ctx.allowUnregisteredDialects();
        mlir::OwningModuleRef module(mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &ctx));
        if (!module) {
            auto message = "DistributedWrapper: Failed to parse source string.\n";
            throw std::runtime_error(message);
        }

        auto *distOp = module->lookupSymbol("dist");
        mlir::FuncOp distFunc;
        if (!(distFunc = llvm::dyn_cast_or_null<mlir::FuncOp>(distOp))) {
            auto message = "DistributedWrapper: MLIR fragment has to contain `dist` FuncOp\n";
            throw std::runtime_error(message);
        }
        auto distFuncTy = distFunc.getType();
        
        // TODO passing a vector<mlir::Type> seems to causes problems...
        // Use enum as work around for now but consider returning mlir::Type
        std::vector<INPUT_TYPE> inputTypes;
        auto distFuncTyArr = distFuncTy.getInputs();
        for (size_t i = 0; i < distFuncTyArr.size(); i++) {
            auto type = distFuncTyArr[i];
            if (type.isIntOrFloat())
                inputTypes.push_back(INPUT_TYPE::Double);
            else
                inputTypes.push_back(INPUT_TYPE::Matrix);
        }
        return inputTypes;
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDWRAPPER_H
