#ifndef SRC_RUNTIME_DISTRIBUTED_MPICOORDINATOR_H
#define SRC_RUNTIME_DISTRIBUTED_MPICOORDINATOR_H

#include <mpi.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/worker/MPISerializer.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <runtime/distributed/worker/MPIWorker.h>
#include <runtime/distributed/worker/MPIHelper.h>

#include <ir/daphneir/Daphne.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/AsmState.h>
##include <mlir/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

#include <vector>



class MPICoordinator{

        public:
        template<class DT>
        static void handleCoordinationPart(DT ***res, size_t numOutputs, const Structure **inputs, size_t numInputs, const char *mlirCode, std::vector<bool> scalars, VectorCombine *combines, DaphneContext *dctx)
        {
            std::vector<WorkerImpl::StoredInfo> outputsStoredInfo;
            std::vector<WorkerImpl::StoredInfo> inputsStoredInfo;
            size_t partitionSize;
            int worldSize= MPIHelper::getCommSize();
            
            //prepare input
            for (size_t i = 0; i < numInputs; i++)
            {
               /* Range range;
                auto combineType = combines[i];
                if (combineType== VectorCombine::ROWS) {
                    partitionSize= inputs[i]->getNumRows()/worldSize;
                    range.r_start = COORDINATOR * partitionSize;
                    range.r_len =partitionSize;
                    range.c_start=0;
                    range.c_len=inputs[i]->getNumCols();
                }
                else if (combineType == VectorCombine::COLS) {
                    partitionSize= inputs[i]->getNumCols()/worldSize;
                    range.c_start = COORDINATOR * partitionSize;
                    range.c_len =partitionSize;
                    range.r_start=0;
                    range.r_len=inputs[i]->getNumRows();

                }*/
                WorkerImpl::StoredInfo info;
                if(!scalars.at(i))
                {
                    Structure* mat = (Structure *)(&inputs[i]);
                    //info = solver.doStore(mat);
                }
                else
                {
                    auto ptr = (double*)(&inputs[i]);
                    double val = *ptr;
                    //info= solver.doStore(&val);
                }
                inputsStoredInfo.push_back(info);

            }
            // prepare output    
            for (size_t i = 0; i < numOutputs; i++)
            {
                auto combineType = combines[i];   
                DistributedData data;
                data.vectorCombine = combineType;
                data.isPlacedAtWorker = true;
                Range range;                                
                if (combineType== VectorCombine::ROWS) {
                    partitionSize = (*res[i])->getNumRows()/worldSize;
                    data.ix  = DistributedIndex(COORDINATOR, 0);                
                    range.r_start = data.ix.getRow() * partitionSize;
                    range.r_len = partitionSize;
                    range.c_start = 0;
                    range.c_len = (*res[i])->getNumCols();
                }
                else if (combineType == VectorCombine::COLS) {
                    partitionSize = (*res[i])->getNumCols()/worldSize;
                    data.ix  = DistributedIndex(0, COORDINATOR);  
                    range.r_start = 0; 
                    range.r_len = (*res[i])->getNumRows(); 
                    range.c_start = data.ix.getCol() * partitionSize;
                    range.c_len = partitionSize;
                }
                std::cout<<"rank "<< COORDINATOR <<" Range rows from "<< range.r_start <<" to " <<( range.r_len + range.r_start)<< " cols from " <<range.c_start <<" to " <<( range.c_len + range.c_start)<<std::endl;
                std::string addr= std::to_string(COORDINATOR);
                // If dp already exists for this worker, update the range and data
                if (auto dp = (*res[i])->getMetaDataObject().getDataPlacementByLocation(addr)) { 
                    (*res[i])->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);
                    dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);                    
                }
                else { // else create new dp entry   
                    AllocationDescriptorMPI allocationDescriptor(
                                            dctx,
                                            COORDINATOR,
                                            data);                                    
                    ((*res[i]))->getMetaDataObject().addDataPlacement(&allocationDescriptor, &range);                    
                } 
            }
            //solver.Compute(&outputsStoredInfo, inputsStoredInfo, mlirCode);

        }
};


#endif
