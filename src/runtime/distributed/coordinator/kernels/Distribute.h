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

#pragma once

#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>

#include <runtime/distributed/worker/MPISerializer.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
struct Distribute {
    static void apply(DT *mat, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void distribute(DT *mat, DCTX(dctx))
{
    Distribute<AT, DT>::apply(mat, dctx);
}


// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************


// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_MPI, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {
        int worldSize;
        MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
        size_t  startRow=0, rowCount=0, startCol=0, colCount=0, remainingRows=0;
        auto partitionSize =  mat->getNumRows()/worldSize;
        remainingRows=mat->getNumRows();
        size_t messageLengths [worldSize];
        void *dataToSend;  
        for(int rank=0;rank<worldSize;rank++)
        {
            startRow= (rank * partitionSize);
            if(rank==worldSize-1){
                    rowCount= remainingRows;
            }
            else{
                rowCount = partitionSize;
            }
            remainingRows-=partitionSize;
            colCount= mat->getNumCols();
            startCol=0;
            Range range;
            range.r_start = startRow;
            range.r_len = rowCount;
            range.c_start = startCol;
            range.c_len = colCount;
            std::string address=std::to_string(rank);
            DataPlacement *dp = mat->getMetaDataObject().getDataPlacementByLocation(address);
            if (dp!=nullptr) {                
                mat->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);     
                auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
                data.ix = DistributedIndex(rank, 0);     
                dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);
            }
            else {
                DistributedData data;
                AllocationDescriptorMPI allocationDescriptor(
                                                dctx,
                                                rank,
                                                data);
                data.ix = DistributedIndex(rank, 0);
                dp = mat->getMetaDataObject().addDataPlacement(&allocationDescriptor, &range);                    
            }
            std::cout<<"rank "<< rank<< " will work on rows from " << startRow << " to "  << startRow+rowCount<<std::endl;
            if (dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
            {
                std::cout<<"worker already has the data"<<std::endl;
                continue;
            }
            MPISerializer::serializeStructure<DT>(&dataToSend, mat ,false, &messageLengths[rank], startRow, rowCount, startCol, colCount);
            MPIWorker::distributeData(messageLengths[rank], dataToSend,rank);
            free(dataToSend);
            long * dataAcknowledgement = (long *) malloc (sizeof(long) * 3);
            std::cout<<"waiting for acknowledgement from "<<rank<<std::endl;
            MPIWorker::getDataAcknowledgementFrom(dataAcknowledgement,rank);
            auto data = dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).getDistributedData();
            data.identifier = dataAcknowledgement[0];
            data.numRows = dataAcknowledgement[1];
            data.numCols = dataAcknowledgement[2];
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorMPI&>(*(dp->allocation)).updateDistributedData(data);
            free(dataAcknowledgement);     
        }

    }
};

// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Distribute<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    static void apply(DT *mat, DCTX(dctx)) {
        struct StoredInfo {
            size_t dp_id;
        }; 
        
        DistributedGRPCCaller<StoredInfo, distributed::Data, distributed::StoredData> caller;
        
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();
    
        assert(mat != nullptr);

        auto r = 0ul;
        for (auto workerIx = 0ul; workerIx < workers.size() && r < mat->getNumRows(); workerIx++) {            
            auto workerAddr = workers.at(workerIx);                      

            auto k = mat->getNumRows() / workers.size();
            auto m = mat->getNumRows() % workers.size();            

            Range range;
            range.r_start = (workerIx * k) + std::min(workerIx, m);
            range.r_len = ((workerIx + 1) * k + std::min(workerIx + 1, m)) - range.r_start;
            range.c_start = 0;
            range.c_len = mat->getNumCols();
                        
            // If dp already exists simply
            // update range (in case we have a different one) and distribute data
            DataPlacement *dp;
            if ((dp = mat->getMetaDataObject().getDataPlacementByLocation(workerAddr))) {                
                mat->getMetaDataObject().updateRangeDataPlacementByID(dp->dp_id, &range);     
                auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
                // TODO Currently we do not support distributing/splitting 
                // by columns. When we do, this should be changed (e.g. Index(0, workerIx))
                data.ix = DistributedIndex(workerIx, 0);
                dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);
            }
            else { // Else, create new object metadata entry
                DistributedData data;
                // TODO Currently we do not support distributing/splitting 
                // by columns. When we do, this should be changed (e.g. Index(0, workerIx))
                data.ix = DistributedIndex(workerIx, 0);
                AllocationDescriptorGRPC allocationDescriptor(
                                            dctx,
                                            workerAddr,
                                            data);
                dp = mat->getMetaDataObject().addDataPlacement(&allocationDescriptor, &range);                    
            }
            // keep track of processed rows
            // Skip if already placed at workers
            if (dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData().isPlacedAtWorker)
                continue;
            distributed::Data protoMsg;
        

            // TODO: We need to handle different data types 
            // (this will be simplified when serialization is implemented)
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix(), 
                                                    range.r_start,
                                                    range.r_start + range.r_len,
                                                    range.c_start,
                                                    range.c_start + range.c_len);

            StoredInfo storedInfo({dp->dp_id}); 
            caller.asyncStoreCall(dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getLocation(), storedInfo, protoMsg);
            r = (workerIx + 1) * k + std::min(workerIx + 1, m);
        }                
                       

        // get results       
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            
            auto storedData = response.result;            

            auto dp = mat->getMetaDataObject().getDataPlacementByID(dp_id);
            
            auto data = dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).getDistributedData();
            data.identifier = storedData.identifier();
            data.numRows = storedData.num_rows();
            data.numCols = storedData.num_cols();
            data.isPlacedAtWorker = true;
            dynamic_cast<AllocationDescriptorGRPC&>(*(dp->allocation)).updateDistributedData(data);            
        }

    }
};

