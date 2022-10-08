#ifndef SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H
#define SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H

#include <mpi.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/worker/MPISerializer.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>


#include <vector>

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST=0, DATASIZE, DATA, DATAACK, MLIRSIZE, MLIR, INPUTKEYS, OUTPUT, OUTPUTKEY,  DETACH
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};


class MPIHelper{

        public:
        static int getCommSize(){
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            return worldSize;    
        }
        
        static WorkerImpl::StoredInfo constructStoredInfo(std::string input)
        {
            WorkerImpl::StoredInfo info;
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
       
        static WorkerImpl::StoredInfo getDataAcknowledgement(int *rank){
            char * dataAcknowledgement;
            size_t len;
            getMessage(rank, DATAACK, MPI_CHAR, (void **)&dataAcknowledgement, &len);
            std::string incomeAck = std::string(dataAcknowledgement);
            WorkerImpl::StoredInfo info=constructStoredInfo(incomeAck);
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