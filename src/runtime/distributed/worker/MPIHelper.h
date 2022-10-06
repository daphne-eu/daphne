#ifndef SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H
#define SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H

#include <mpi.h>
//#include "runtime/distributed/worker/MPISerializer.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/distributed/worker/MPISerializer.h"
#include <runtime/distributed/worker/WorkerImpl.h>
#include <unistd.h>
#include  <iostream>
#include<sstream>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <runtime/distributed/worker/MPIWorker.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#include <ir/daphneir/Daphne.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

#include <vector>

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST=0, DATASIZE, DATA, DATAACK, MLIRSIZE, MLIR, INPUTKEYS, OUTPUT, OUTPUTKEY,  DETACH
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};

struct StoredInfo {
    std::string identifier;
    size_t numRows, numCols;
    std::string toString(){
        return identifier+","+std::to_string(numRows)+","+std::to_string(numCols);
    }
};

class MPIHelper{
    public:
        //Utility functions will be used by the coordinator
        static int getCommSize(){
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            return worldSize;    
        }
       
        template<class DT>
        static void handleCoordinationPart(DT ***res, size_t numOutputs, const Structure **inputs, size_t numInputs, const char *mlirCode, VectorCombine *combines, DaphneContext *dctx)
        {
            size_t partitionSize;
            int worldSize= getCommSize();    
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
               // solver.Compute()
                //task.set_mlir_code(mlirCode);
                //MPISerializer::serializeTask(&taskToSend, &messageLengths[rank], &task);
                //MPIWorker::distributeTask(messageLengths[rank], taskToSend,rank);
                //free(taskToSend);
            }

        }
      
        static StoredInfo constructStoredInfo(std::string input)
        {
            StoredInfo info;
            std::stringstream s_stream(input);
            std::vector<std::string> results;
            while(s_stream.good()) {
                std::string substr;
                getline(s_stream, substr, ','); //get first string delimited by comma
                results.push_back(substr);
            }
            info.identifier=results.at(0);
            sscanf(results.at(1).c_str() , "%zu", &info.numRows);
            sscanf(results.at(2).c_str() , "%zu", &info.numCols);
            return info; 
        }
       
        static distributed::Data getResults(int *rank){
            size_t resultsLen=0;
            void * results;
            distributed::Data matProto;
            getMessage(rank, OUTPUT, MPI_UNSIGNED_CHAR ,&results, &resultsLen);
            MPISerializer::deserializeStructure(&matProto, results , resultsLen);
           // std::cout<<"got results from "<<*rank<<std::endl;     
            free(results);
            return matProto;
        }
       
        static StoredInfo getDataAcknowledgement(int *rank){
            char * dataAcknowledgement;
            size_t len;
            getMessage(rank, DATAACK, MPI_CHAR, (void **)&dataAcknowledgement, &len);
            std::string incomeAck = std::string(dataAcknowledgement);
            StoredInfo info=constructStoredInfo(incomeAck);
            free(dataAcknowledgement);
            return info;  
        }
       
        static void sendData(size_t messageLength, void * data){
            int worldSize=getCommSize();
            int  message= messageLength;
            for(int rank=0; rank<worldSize;rank++)
            {
                if(rank==COORDINATOR)
                    continue;          
                MPI_Send(&message,1, MPI_INT, rank, BROADCAST, MPI_COMM_WORLD);                    
            }
            MPI_Bcast(data, message, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
        }
       
        static void distributeData(size_t messageLength, void * data, int rank){
            distributeWithTag(DATA, messageLength, data, rank);
        }
       
        static void distributeTask(size_t messageLength, void * data, int rank){
            distributeWithTag(MLIR, messageLength, data, rank);
        }
       
        static void displayDataStructure(Structure * inputStruct, std::string dataToDisplay)
        {
            DenseMatrix<double> *res= dynamic_cast<DenseMatrix<double>*>(inputStruct);
            double * allValues = res->getValues();
                for(size_t r = 0; r < res->getNumRows(); r++){
                    for(size_t c = 0; c < res->getNumCols(); c++){
                        dataToDisplay += std::to_string(allValues[c]) + " , " ;
                    }
                    dataToDisplay+= "\n";
                    allValues += res->getRowSkip();
                }
                std::cout<<dataToDisplay<<std::endl;
        }
        
        static void displayData(distributed::Data data, int rank)
        {
            std::string dataToDisplay="rank "+ std::to_string(rank) + " got ";
            if(data.matrix().matrix_case()){
                const distributed::Matrix& mat = data.matrix();
                dataToDisplay += "matrix :";
                auto temp= DataObjectFactory::create<DenseMatrix<double>>(data.mutable_matrix()->num_rows(), data.mutable_matrix()->num_cols(), false);
                DenseMatrix<double> *res =  dynamic_cast<DenseMatrix<double> *>(temp);
                ProtoDataConverter<DenseMatrix<double>>::convertFromProto(mat, res);
                displayDataStructure(res, dataToDisplay);
                
            }
            else
            {
                dataToDisplay += " scalar  lf:";
                dataToDisplay += std::to_string(data.value().f64());
                std::cout<<dataToDisplay<<std::endl;

            }
        }

        static void getMessage(int * rank, int tag, MPI_Datatype type, void ** data, size_t * len)
        {
            int size;
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, type, &size);
            *rank=status.MPI_SOURCE;
            if(type==MPI_UNSIGNED_CHAR)
            {
                *data = malloc(size * sizeof(unsigned char)); 
            }
            else if(type==MPI_CHAR)
            {
                *data = malloc(size * sizeof(char)); 
            }
            MPI_Recv(*data, size, type, status.MPI_SOURCE , tag, MPI_COMM_WORLD, &status);
            *len=size;
        }
       
        static void getMessageFrom(int rank, int tag, MPI_Datatype type, void ** data, size_t * len)
        {
            int size;
            MPI_Status status;
            MPI_Probe(rank, tag, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, type, &size);
            if(type==MPI_UNSIGNED_CHAR)
            {
                *data = malloc(size * sizeof(unsigned char)); 
            }
            else if(type==MPI_CHAR)
            {
                *data = malloc(size * sizeof(char)); 
            }

            MPI_Recv(*data,size, type, status.MPI_SOURCE , tag, MPI_COMM_WORLD, &status);
            *len=size;
        }
        
        static void distributeWithTag (TypesOfMessages tag, size_t messageLength, void * data, int rank)
        {
            if(rank == COORDINATOR)
                return;
            int message = messageLength;
            int sizeTag=-1, dataTag=-1;
           // std::cout<<"message size is "<< message << " tag "<< tag <<std::endl;
            switch(tag)
            {
                case DATA:
                    sizeTag = DATASIZE;
                    dataTag = DATA;
                break;
                case MLIR:
                    sizeTag = MLIRSIZE;
                    dataTag = MLIR;
                default:
                break;
            }
           // std::cout<<"message size is "<< message << " tag "<< sizeTag <<std::endl;
            MPI_Send(&message,1, MPI_INT, rank, sizeTag, MPI_COMM_WORLD);                    
            MPI_Send(data, message, MPI_UNSIGNED_CHAR, rank, dataTag ,MPI_COMM_WORLD);
        }
        
};

#endif