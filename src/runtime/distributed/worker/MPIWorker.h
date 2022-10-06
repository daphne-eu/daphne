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
#include <runtime/distributed/worker/WorkerImpl.h>
#include <unistd.h>
#include  <iostream>
#include<sstream>

#include <ir/daphneir/Daphne.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Parser.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>
#include <vector>

using mlir::daphne::VectorSplit;
using mlir::daphne::VectorCombine;

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST=0, DATASIZE, DATA, DATAACK, MLIRSIZE, MLIR, INPUTKEYS, OUTPUT, OUTPUTKEY,  DETACH
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};
class MPIWorker: WorkerImpl {
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
        std::vector<StoredInfo> inputs;
        
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
        
        StoredInfo updateInputs (distributed::Data * message, void * data, int messageLength)
        {
            StoredInfo info;
            MPISerializer::deserializeStructure(message, data, messageLength);
            if(message->matrix().matrix_case()){
                auto matrix = &message->matrix();
                DenseMatrix<double> *mat= DataObjectFactory::create<DenseMatrix<double>>(message->mutable_matrix()->num_rows(), message->mutable_matrix()->num_cols(), false);
                //Structure *res =  dynamic_cast<DenseMatrix<double> *>(temp);
                ProtoDataConverter<DenseMatrix<double>>::convertFromProto(*matrix, mat);
                info = this->Store<Structure>(mat);
            }
            else
            {
                double val= message->value().f64();
                info= this->Store(&val);
            }
            inputs.push_back(info);
           // std::cout<<"id "<<id<<" added something " << inputs.size()<<std::endl;
            return info; 
        }
        
        void sendResult(std::vector<StoredInfo> outputs)
        {
            for(int i=0;i<outputs.size();i++)
            {
                StoredInfo tempInfo=outputs.at(i);
                Structure * res =Transfer(tempInfo);
                void * dataToSend;
                size_t messageLength;
                MPISerializer::serializeStructure<Structure>(&dataToSend, res, false, &messageLength); 
                int  len= messageLength;
                MPI_Send(dataToSend, len, MPI_UNSIGNED_CHAR, COORDINATOR, OUTPUT, MPI_COMM_WORLD);
              //  std::string message = "result from ("+ std::to_string(id) +") is:\n";
               // displayDataStructure(res, message);
                free(dataToSend);
            }

        }
       
        void prepareBufferForMessage(void ** data, int * messageLength, MPI_Datatype type, int source, int tag)
        {
            MPI_Status messageStatus;
            MPI_Recv(messageLength, 1, type, source, tag, MPI_COMM_WORLD, &messageStatus);
           // std::cout<< id<<" in distribute size " <<*messageLength << " tag " << tag <<std::endl;
            *data = malloc(*messageLength * sizeof(unsigned char));
        } 
        
        void sendDataACK(StoredInfo info)
        {
            //std::cout<< "ack from " << id << " will have " << info.identifier << " , " << std::to_string(info.numRows) << " , " <<std::to_string(info.numCols)<<std::endl;
            std::string toSend= info.toString();
            MPI_Send(toSend.c_str(), sizeof(toSend), MPI_CHAR, COORDINATOR, DATAACK, MPI_COMM_WORLD);
            //std::cout<<"rank " << id <<" acknowledging" <<std::endl;
        }
        
        void detachFromComputingTeam(){
            myState = DETACHED;
            //std::cout<<"I am " << id <<". I got detach message... " << std::endl;
        }
        
        void terminate(){
            myState = TERMINATED;
           // std::cout<<"I am worker " << id << ". I'll rest in peace" << std::endl;
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
            double val;
            distributed::Data protoMsgData;
            distributed::Task protoMsgTask;
            std::string printData="";
            size_t index=0, rows=0, cols=0;
            StoredInfo info;
            std::vector<StoredInfo> outputs;
            WorkerImpl::Status exStatus(true);
            switch(tag){
                case BROADCAST:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, BROADCAST);
                    MPI_Bcast(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    info=updateInputs(&protoMsgData, data, messageLength);
                    //displayData(protoMsgData, id);
                    free(data);
                    sendDataACK(info);
                break;

                case DATASIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, DATASIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, DATA,MPI_COMM_WORLD, &messageStatus);
                    info=updateInputs(&protoMsgData, data, messageLength);
                    //std::cout<<"rank " << id <<"will send ack" <<std::endl;
                    //displayData(protoMsgData,id);
                    free(data);
                    sendDataACK(info);
                    //std::cout<<"rank " << id <<" sent ack" <<std::endl;
                break;

                case MLIRSIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, MLIRSIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MLIR,MPI_COMM_WORLD, &messageStatus);
                    protoMsgTask.ParseFromArray(data, messageLength);
                    //printData = "worker "+std::to_string(id)+" got MLIR "+protoMsgTask.mlir_code();
                    exStatus=this->Compute(&outputs, inputs, protoMsgTask.mlir_code());
                    //std::cout<<printData<<std::endl;
                    if(!(exStatus.ok()))
                        std::cout<<"error!";    
                    // std::cout<<"computation is done"<<std::endl;
                    sendResult(outputs);
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
