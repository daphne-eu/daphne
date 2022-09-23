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
    BROADCAST=0, DISTRIBUTE, DETACH, DATA, MLIR, DISTRIBUTEDATA
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};
class MPIWorker{
    public:
        //Utility functions will be used by the coordinator
        static void sendData(size_t messageLength, void * data){
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            int  message= messageLength;
            for(int rank=0; rank<worldSize;rank++)
            {
                if(rank==COORDINATOR)
                    continue;     
                MPI_Send(&message,1, MPI_INT, rank, BROADCAST, MPI_COMM_WORLD);                    
            }
            MPI_Bcast(data, message, MPI_BYTE, COORDINATOR, MPI_COMM_WORLD);
            /*int messageSize = rows*columns;
            void * buffer = (void *) data;
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            for(int rank=0; rank<worldSize;rank++)
            {
                if(rank==COORDINATOR)
                    continue;   
                int buf [] = {rows, columns};    
                MPI_Send(buf,2, MPI_INT, rank, BROADCAST, MPI_COMM_WORLD);                    
            }
            MPI_Bcast(buffer, messageSize, dataType, COORDINATOR, MPI_COMM_WORLD);
            */
        }
        
        static void distributeData(size_t messageLength, void * data, int rank){
            int message = messageLength;
            MPI_Send(&message,1, MPI_INT, rank, DISTRIBUTE, MPI_COMM_WORLD);                    
            MPI_Send(data, message, MPI_BYTE, rank, DISTRIBUTEDATA ,MPI_COMM_WORLD);

            /*std::cout<<"coordinator will send data "<<std::endl;
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            int workerPartition = (rows/worldSize) * columns; // row partition
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            for(int rank=1; rank<worldSize;rank++)
            {  
                int buf [] = {rows/worldSize, columns};    
                MPI_Send(buf,2, MPI_INT, rank, DISTRIBUTE, MPI_COMM_WORLD);                    
            }
            MPI_Request requests[worldSize-1];
            MPI_Status statuses [worldSize-1];
            for(int rank=1;rank<worldSize;rank++)
            {
                MPI_Isend(data, workerPartition, MPI_DOUBLE, rank, DISTRIBUTEDATA ,MPI_COMM_WORLD, &requests[rank-1]);
            }
            MPI_Waitall(worldSize-1, requests, statuses);*/
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
        //std::vector<DenseMatrix<double>*> inputs;
        void detachFromComputingTeam(){
            myState = DETACHED;
            std::cout<<"I am " << id <<". I got detach message... " << std::endl;
        }
        void terminate(){
            myState = TERMINATED;
            std::cout<<"I am worker " << id << ". I'll rest in peace" << std::endl;
        }
        void continueComputing(){
        }
        void handleInCommingMessages(MPI_Status status){
            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;
            int size;
            int dataSize;
            int codeSize; 
            MPI_Status messageStatus;
            unsigned char  * info;
            double * data;
            char * mlirCode;
            int messageLength;
            DenseMatrix<double> *mat=nullptr;
            distributed::Data protoMsg;
            switch(tag){
                case BROADCAST:
                    MPI_Recv(&messageLength, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &messageStatus);
                    data = (double*) malloc(messageLength * sizeof(double));
                    MPI_Bcast(data, messageLength, MPI_BYTE, COORDINATOR, MPI_COMM_WORLD);
                    protoMsg.ParseFromArray(data, messageLength);
                    mat= MPISerializer<DenseMatrix<double>>::deserialize(data, messageLength);
                    std::cout<<"rank "  << id << " broadcast message message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    protoMsgs.push_back(protoMsg);
                    free(data);
                break;

                case DISTRIBUTE:
                    MPI_Recv(&messageLength, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &messageStatus);
                    data = (double*) malloc(messageLength * sizeof(double));
                    MPI_Status status;
                    MPI_Recv(data, messageLength, MPI_BYTE, COORDINATOR, DISTRIBUTEDATA,MPI_COMM_WORLD, &status);
                    protoMsg.ParseFromArray(data, messageLength);
                    protoMsgs.push_back(protoMsg);
                    mat= MPISerializer<DenseMatrix<double>>::deserialize(data, messageLength);
                    std::cout<<"rank "  << id << " distribute message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    free(data);
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
