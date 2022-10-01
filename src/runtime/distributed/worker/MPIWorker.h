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

#ifndef SRC_RUNTIME_DISTRIBUTED_MPIWORKER_H
#define SRC_RUNTIME_DISTRIBUTED_MPIWORKER_H

#include <mpi.h>
//#include "runtime/distributed/worker/MPISerializer.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/distributed/worker/MPISerializer.h"
#include <unistd.h>
#include  <iostream>

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST=0, DISTRIBUTE, DETACH, DATA, MLIR, DISTRIBUTEDATA, DATAACK
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};
class MPIWorker{
    public:
        //Utility functions will be used by the coordinator
        static int getCommSize(){
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            return worldSize;    
        }
        static int getDataAcknowledgementFrom (long * dataAcknowledgement, int rank){
            MPI_Status status;
            if(rank==COORDINATOR)
            {
                std::cout<<"coordinator does not need to ack receive it owns the data" <<std::endl;
                return 0;
            }
            if(rank==-1)
                rank = MPI_ANY_SOURCE;
            MPI_Recv(dataAcknowledgement,3, MPI_LONG, MPI_ANY_SOURCE , DATAACK, MPI_COMM_WORLD, &status);
            return status.MPI_SOURCE;
        }
        static int getDataAcknowledgement (long * dataAcknowledgement){
            getDataAcknowledgementFrom(dataAcknowledgement, -1);
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
            if(rank == COORDINATOR)
                return;
            int message = messageLength;
            MPI_Send(&message,1, MPI_INT, rank, DISTRIBUTE, MPI_COMM_WORLD);                    
            MPI_Send(data, message, MPI_UNSIGNED_CHAR, rank, DISTRIBUTEDATA ,MPI_COMM_WORLD);
        }
        void displayData(DenseMatrix<double> * mat)
        {
            std::string dataToDisplay="rank "+ std::to_string(id) + " got:\n";
            size_t numRows = mat->getNumRows();
            size_t numCols = mat->getNumCols();

            double * allValues = mat->getValues();
            for(size_t r = 0; r < numRows; r++){
                for(size_t c = 0; c < numCols; c++){
                    dataToDisplay += std::to_string(allValues[c]) + " , " ;
                }
                dataToDisplay+= "\n";
                allValues += mat->getRowSkip();
            }
            std::cout<<dataToDisplay<<std::endl;
        }
        MPIWorker(){//TODO
            MPI_Comm_rank(MPI_COMM_WORLD, &id);
        }
        
        ~MPIWorker(){//TODO
        }
        
        void joinComputingTeam(){
            int inCommingMessage=0;
            MPI_Status status;
            while(myState!=TERMINATED){//
                MPI_Iprobe(COORDINATOR, MPI_ANY_TAG, MPI_COMM_WORLD, &inCommingMessage, &status);
                if(inCommingMessage && myState!=DETACHED)
                {
                    handleInCommingMessages(status);
                }
                else
                    continueComputing(); // takes form a queue // hocks for scheuling
            }
        }
    private:
        int id;
        int myState=LISTENING;
        int temp=0;
        std::vector<distributed::Data> protoMsgs;
        void sendDataACK(long index, long rows, long cols)
        {
            long dataAcknowledgement [3];
            dataAcknowledgement[0]= index;
            dataAcknowledgement[1] = rows;
            dataAcknowledgement[2] = cols;
            MPI_Send(dataAcknowledgement, 3, MPI_LONG, COORDINATOR, DATAACK, MPI_COMM_WORLD);
        }
        void detachFromComputingTeam(){
            myState = DETACHED;
            std::cout<<"I am " << id <<". I got detach message... " << std::endl;
        }
        void terminate(){
            myState = TERMINATED;
            std::cout<<"I am worker " << id << ". I'll rest in peace" << std::endl;
        }
        void continueComputing(){
            if(myState==DETACHED)
                myState=TERMINATED;
        }
        void handleInCommingMessages(MPI_Status status){
            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;
            int size;
            int dataSize;
            int codeSize; 
            MPI_Status messageStatus;
            unsigned char  * info;
            unsigned char * data;
            char * mlirCode;
            int messageLength;
            DenseMatrix<double> *mat=nullptr;
            distributed::Data protoMsg;
            switch(tag){
                case BROADCAST:
                    MPI_Recv(&messageLength, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &messageStatus);
                    std::cout<<"in broadcast received "<< messageLength <<std::endl; 
                    data = (unsigned char*) malloc(messageLength * sizeof(unsigned char));
                    MPI_Bcast(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    std::cout<<"in broadcast received data "<<std::endl;
                    protoMsg.ParseFromArray(data, messageLength);
                    mat= MPISerializer<DenseMatrix<double>>::deserialize(data, messageLength);
                    //std::cout<<"rank "  << id << " broadcast message message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    //displayData(mat);
                    protoMsgs.push_back(protoMsg);
                    free(data);
                    sendDataACK(protoMsgs.size()-1, mat->getNumRows(), mat->getNumCols());
                break;

                case DISTRIBUTE:
                    MPI_Recv(&messageLength, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &messageStatus);
                    //std::cout<<"allocated size " <<messageLength;
                    data = (unsigned char*) malloc(messageLength * sizeof(unsigned char));
                    MPI_Status status;
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, DISTRIBUTEDATA,MPI_COMM_WORLD, &status);
                    protoMsg.ParseFromArray(data, messageLength);
                    protoMsgs.push_back(protoMsg);
                    mat= MPISerializer<DenseMatrix<double>>::deserialize(data, messageLength);
                    displayData(mat);
                    std::cout<<"rank "  << id << " distribute message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    free(data);
                    sendDataACK(protoMsgs.size()-1, mat->getNumRows(), mat->getNumCols());
                break;

                case MLIR:
                   /* info = (unsigned char *) malloc(size * sizeof(unsigned char));
                    MPI_Recv(info, size, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &messageStatus);
                    MPISerializer::getSizeOfCode(info, &codeSize);
                    mlirCode = (char *) malloc(codeSize * sizeof(char));
                    MPI_Bcast(mlirCode, codeSize, MPI_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    printf("=======\nI am %d got the following MLIR code\n%s\n", id, mlirCode);
                    free(info);
                    free(mlirCode);*/
                break;

                case DETACH:
                    unsigned char terminateMessage;
                    MPI_Recv(&terminateMessage, 1, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &messageStatus);
                    detachFromComputingTeam();
                break;

                default:
                    //TODO
                break;
            }
        }
};

#endif
