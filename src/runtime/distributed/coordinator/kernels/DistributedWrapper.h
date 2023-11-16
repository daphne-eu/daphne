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

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#ifdef USE_MPI
    #include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#endif

#include <mlir/InitAllDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>
#include <vector>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

template <class DT>
class DistributedWrapper {
private:
    DCTX(_dctx);

protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, const Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }
public:
    DistributedWrapper(DCTX(dctx)) : _dctx(dctx) {
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
        auto ctx = DistributedContext::get(_dctx);
        auto workers = ctx->getWorkers();
        
        // Backend Implementation 
        // gRPC hard-coded selection
        const auto allocation_type=_dctx->getUserConfig().distributedBackEndSetup;
        //std::cout<<"Distributed wrapper " <<std::endl;
        // output allocation for row-wise combine
        for(size_t i = 0; i < numOutputs; ++i) {
            if(*(res[i]) == nullptr && outRows[i] != -1 && outCols[i] != -1) {
                auto zeroOut = combines[i] == mlir::daphne::VectorCombine::ADD;
                // TODO we know result is only DenseMatrix<double> for now,
                // but in the future this will change to support other DataTypes
                *(res[i]) = DataObjectFactory::create<DT>(outRows[i], outCols[i], zeroOut);
            }
        }

        // Currently an input might appear twice in the inputs array of a pipeline.
        // E.g. an input is needed both "Distributed/Scattered" and "Broadcasted".
        // This might cause conflicts regarding the meta data of an object since 
        // we need to represent multiple different "DataPlacements" (ways of how the data is distributed).
        // A solution would be to support multiple meta data for a single Structure, each one representing
        // a different way data is placed.
        // @pdamme suggested that if an input is appearing multiple times in the input pipeline, 
        // we can probably solve this by applying a more "aggressive" pipelining and removing duplicate inputs.
        // For now we support only unique inputs.
        std::vector<const Structure *> vec;
        for (size_t i = 0; i < numInputs; i++)
            vec.push_back(inputs[i]);
        sort(vec.begin(), vec.end());
        const bool hasDuplicates = std::adjacent_find(vec.begin(), vec.end()) != vec.end();
        if(hasDuplicates)
            throw std::runtime_error("Distributed runtime only supports unique inputs for now (no duplicate inputs in a pipeline)");
        
        // Parse mlir code fragment to determin pipeline inputs/outputs
        auto inputTypes = getPipelineInputTypes(mlirCode);
        std::vector<bool> scalars;
        for(auto t : inputTypes)
        {
            scalars.push_back(t!=INPUT_TYPE::Matrix);
        }
        // Distribute and broadcast inputs        
        // Each primitive sends information to workers and changes the Structures' metadata information 
        for (auto i = 0u; i < numInputs; ++i) {
            // if already placed on workers, skip
            // TODO maybe this is not enough. We might also need to check if data resides in the specific way we need to.
            // (i.e. rows/cols splitted accordingly). If it does then we can skip.
            if (isBroadcast(splits[i], inputs[i])){
                auto type = inputTypes.at(i);
                if (type==INPUT_TYPE::Matrix) {
                    if(allocation_type==ALLOCATION_TYPE::DIST_MPI){
#ifdef USE_MPI           
                        broadcast<ALLOCATION_TYPE::DIST_MPI>(inputs[i], false, _dctx);
#endif
                    }
                    else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) 
                    { 
                        broadcast<ALLOCATION_TYPE::DIST_GRPC_ASYNC>(inputs[i], false, _dctx);
                    }
                    else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) 
                    { 
                        broadcast<ALLOCATION_TYPE::DIST_GRPC_SYNC>(inputs[i], false, _dctx);
                    }
                }
                else {
                        if(allocation_type==ALLOCATION_TYPE::DIST_MPI){
#ifdef USE_MPI 
                            broadcast<ALLOCATION_TYPE::DIST_MPI>(inputs[i], true, _dctx);
#endif
                        }
                        else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) 
                        { 
                            broadcast<ALLOCATION_TYPE::DIST_GRPC_ASYNC>(inputs[i], true, _dctx);
                        }
                        else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) 
                        { 
                            broadcast<ALLOCATION_TYPE::DIST_GRPC_SYNC>(inputs[i], true, _dctx);
                        }
                }
            }
            else {
                assert(splits[i] == VectorSplit::ROWS && "only row split supported for now");
                // std::cout << i << " distr: " << inputs[i]->getNumRows() << " x " << inputs[i]->getNumCols() << std::endl;
                if(allocation_type==ALLOCATION_TYPE::DIST_MPI){
#ifdef USE_MPI 
                    distribute<ALLOCATION_TYPE::DIST_MPI>(inputs[i], _dctx);
#endif
                }
                else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) 
                {
                    distribute<ALLOCATION_TYPE::DIST_GRPC_ASYNC>(inputs[i], _dctx);  
                }
                else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) 
                { 
                    distribute<ALLOCATION_TYPE::DIST_GRPC_SYNC>(inputs[i], _dctx);  
                }
            }
        }

        if(allocation_type==ALLOCATION_TYPE::DIST_MPI){
#ifdef USE_MPI   
            distributedCompute<ALLOCATION_TYPE::DIST_MPI>(res, numOutputs, inputs, numInputs, mlirCode, combines, _dctx);
#endif        
        }
        else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) { 
            distributedCompute<ALLOCATION_TYPE::DIST_GRPC_ASYNC>(res, numOutputs, inputs, numInputs, mlirCode, combines, _dctx);   
        }
        else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) { 
            distributedCompute<ALLOCATION_TYPE::DIST_GRPC_SYNC>(res, numOutputs, inputs, numInputs, mlirCode, combines, _dctx);   
        }
        //handle my part as coordinator we currently exclude the coordinator
        /*if(alloc_type==ALLOCATION_TYPE::DIST_MPI)
        {
            bool isScalar=true;
            MPICoordinator::handleCoordinationPart<DT>(res, numOutputs, inputs, numInputs, mlirCode, scalars , combines, _dctx);
        }*/

        // Collect
        for (size_t o = 0; o < numOutputs; o++){
            assert ((combines[o] == VectorCombine::ROWS || combines[o] == VectorCombine::COLS) && "we only support rows/cols combine atm");
            if(allocation_type==ALLOCATION_TYPE::DIST_MPI){
#ifdef USE_MPI 
                distributedCollect<ALLOCATION_TYPE::DIST_MPI>(*res[o], _dctx);      
#endif
            }
            else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) 
            { 
                distributedCollect<ALLOCATION_TYPE::DIST_GRPC_ASYNC>(*res[o], _dctx);
            }
            else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) 
            { 
                distributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC>(*res[o], _dctx);
            }
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
        ctx.getOrLoadDialect<mlir::func::FuncDialect>();
        ctx.allowUnregisteredDialects();
        //std::cout<<mlirCode<<std::endl;
        mlir::OwningOpRef<mlir::ModuleOp> module(mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &ctx));
        if (!module) {
            auto message = "DistributedWrapper: Failed to parse source string.\n";
            throw std::runtime_error(message);
        }
        auto *distOp = module->lookupSymbol("dist");
        mlir::func::FuncOp distFunc;
        if (!(distFunc = llvm::dyn_cast_or_null<mlir::func::FuncOp>(distOp))) {
            auto message = "DistributedWrapper: MLIR fragment has to contain `dist` FuncOp\n";
            throw std::runtime_error(message);
        }
        auto distFuncTy = distFunc.getFunctionType();
        
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
