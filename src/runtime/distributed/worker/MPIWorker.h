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
    BROADCAST=0, DATASIZE=1, DATA=2, DATAACK=3, MLIRSIZE=4, MLIR=5, DETACH=6
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
        static int getDataAcknowledgementFrom (size_t * dataAcknowledgement, int rank){
            MPI_Status status;
            if(rank==COORDINATOR)
            {
                std::cout<<"coordinator does not need to ack receive it owns the data" <<std::endl;
                dataAcknowledgement[0]=0;
                dataAcknowledgement[1]=0;
                dataAcknowledgement[2]=0;
                return 0;
            }
            if(rank==-1)
                rank = MPI_ANY_SOURCE;
            MPI_Recv(dataAcknowledgement,3, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE , DATAACK, MPI_COMM_WORLD, &status);
            return status.MPI_SOURCE;
        }
        static int getDataAcknowledgement (size_t * dataAcknowledgement){
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
            distributeWithTag(DATA, messageLength, data, rank);
        }
        static void distributeTask(size_t messageLength, void * data, int rank){
            distributeWithTag(MLIR, messageLength, data, rank);
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
        void prepareBufferForMessage(void ** data, int * messageLength, MPI_Datatype type, int source, int tag)
        {
            MPI_Status messageStatus;
            MPI_Recv(messageLength, 1, type, source, tag, MPI_COMM_WORLD, &messageStatus);
           // std::cout<< id<<" in distribute size " <<*messageLength << " tag " << tag <<std::endl;
            *data = malloc(*messageLength * sizeof(unsigned char));
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
        void sendDataACK(size_t index, size_t rows, size_t cols)
        {
            size_t *dataAcknowledgement = (size_t *)malloc(sizeof(size_t)*3);
            dataAcknowledgement[0]= index;
            dataAcknowledgement[1] = rows;
            dataAcknowledgement[2] = cols;
            MPI_Send(dataAcknowledgement, 3, MPI_UNSIGNED_LONG, COORDINATOR, DATAACK, MPI_COMM_WORLD);
            free(dataAcknowledgement);
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
            MPI_Status messageStatus;
            void * data;
            char * mlirCode;
            int messageLength;
            DenseMatrix<double> *mat=nullptr;
            distributed::Data protoMsgData;
            distributed::Task protoMsgTask;
            std::string printData="";
            size_t index=0, rows=0, cols=0;
            switch(tag){
                case BROADCAST:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, BROADCAST);
                    MPI_Bcast(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    //std::cout<<"in broadcast received data "<<std::endl;
                    protoMsgData.ParseFromArray(data, messageLength);
                    mat= MPISerializer::deserializeStructure<DenseMatrix<double>>(data, messageLength);
                    //std::cout<<"rank "  << id << " broadcast message message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    //displayData(mat);
                    index= protoMsgs.size();
                    rows=mat->getNumRows();
                    cols= mat->getNumCols();
                   // std::cout<<"rank "<<id<<" stored messages " <<index <<std::endl;
                    protoMsgs.push_back(protoMsgData);
                    free(data);
                    sendDataACK(index,rows ,cols );
                break;

                case DATASIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, DATASIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, DATA,MPI_COMM_WORLD, &messageStatus);
                    protoMsgData.ParseFromArray(data, messageLength);
                    protoMsgs.push_back(protoMsgData);
                    mat= MPISerializer::deserializeStructure<DenseMatrix<double>>(data, messageLength);
                    index= protoMsgs.size();
                    rows=mat->getNumRows();
                    cols= mat->getNumCols();
                   // std::cout<<"rank "<<id<<" stored messages " <<index <<std::endl;
                    displayData(mat);
                  //  std::cout<<"rank "  << id << " distribute message size "<<messageLength<< " got rows "<< mat->getNumRows()  << " got cols "<< mat->getNumCols()<<std::endl ;
                    free(data);
                    sendDataACK(index,rows, cols);
                break;

                case MLIRSIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, MLIRSIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MLIR,MPI_COMM_WORLD, &messageStatus);
                    protoMsgTask.ParseFromArray(data, messageLength);
                    printData = "worker "+std::to_string(id)+" got MLIR "+protoMsgTask.mlir_code();
                    std::cout<<printData<<std::endl;
                    free(data);
                break;

                case DETACH:
                    unsigned char terminateMessage;
                    MPI_Recv(&terminateMessage, 1, MPI_UNSIGNED_CHAR, source, DETACH, MPI_COMM_WORLD, &messageStatus);
                    detachFromComputingTeam();
                break;

                default:
                    //TODO
                break;
            }
        }
};

#endif
