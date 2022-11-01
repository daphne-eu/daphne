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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/worker/MPISerializer.h>

#include <runtime/distributed/worker/WorkerImpl.h>
#include <runtime/distributed/worker/MPIHelper.h>

#include <unistd.h>
#include <iostream>
#include <sstream>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>

class MPIWorker: WorkerImpl {
    public:
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
        
        WorkerImpl::Status doCompute(std::vector<StoredInfo> * outputsStoredInfo, std::vector<StoredInfo> inputsStoredInfo, std::string mlirCode)
        {
            return this->Compute(outputsStoredInfo, inputsStoredInfo, mlirCode);
        }
        StoredInfo doStore(Structure *mat)
        {
                return this->Store(mat);
        }
        StoredInfo doStore(double *val)
        {
              return  this->Store(val);
        }
    private:
        int id;
        int myState=LISTENING;
        int temp=0;
        std::vector<StoredInfo> allReceivedInputs;
        std::vector<std::string> currentPipelineIdentifiers;
        void getCurrentPipelineInputs(std::vector<StoredInfo> *currentPipelineInputs, std::vector<std::string> currentPipelineIdentifiers)
        {
            for(size_t i=0;i<currentPipelineIdentifiers.size();i++)
            {
                for(size_t j=0;j< allReceivedInputs.size();j++)
                {
                    if(currentPipelineIdentifiers.at(i)==allReceivedInputs.at(j).identifier)
                    {
                        currentPipelineInputs->push_back(allReceivedInputs.at(j));
                        break;
                    }
                }
            }
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
                info = this->doStore(mat);
            }
            else
            {
                double val= message->value().f64();
                info= this->doStore(&val);
            }
            allReceivedInputs.push_back(info);
            std::cout<<"input object has been received and identifier "<<info.identifier<<" has been added at " << id<<std::endl;
            currentPipelineIdentifiers.push_back(info.identifier);
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
            std::vector<StoredInfo> currentPipelineInputs;
            std::string identifier;
            WorkerImpl::Status exStatus(true);
            switch(tag){
                case BROADCAST:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, BROADCAST);
                    MPI_Bcast(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    info=updateInputs(&protoMsgData, data, messageLength);
                    //MPIHelper::displayData(protoMsgData, id);
                    free(data);
                    sendDataACK(info);
                break;
                case OBJECTIDENTIFIERSIZE:
                    std::cout<<"Identifier Message "<<std::endl;
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, OBJECTIDENTIFIERSIZE);
                    MPI_Recv(data, messageLength, MPI_CHAR, COORDINATOR, OBJECTIDENTIFIER,MPI_COMM_WORLD, &messageStatus);
                    identifier = std::string((const char *) data);
                    std::cout<<"identifier "<<identifier <<" received at "<< id<<std::endl;
                    currentPipelineIdentifiers.push_back(identifier);
                break;
                case DATASIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, DATASIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, DATA,MPI_COMM_WORLD, &messageStatus);
                    info=updateInputs(&protoMsgData, data, messageLength);
                    //std::cout<<"rank " << id <<"will send ack" <<std::endl;
                    //MPIHelper::displayData(protoMsgData,id);
                    free(data);
                    sendDataACK(info);
                    //std::cout<<"rank " << id <<" sent ack" <<std::endl;
                break;

                case MLIRSIZE:
                    prepareBufferForMessage(&data, &messageLength, MPI_INT, source, MLIRSIZE);
                    MPI_Recv(data, messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MLIR,MPI_COMM_WORLD, &messageStatus);
                    protoMsgTask.ParseFromArray(data, messageLength);
                    //printData = "worker "+std::to_string(id)+" got MLIR "+protoMsgTask.mlir_code();
                    getCurrentPipelineInputs(&currentPipelineInputs, currentPipelineIdentifiers);
                    exStatus=this->doCompute(&outputs,currentPipelineInputs , protoMsgTask.mlir_code());
                    //std::cout<<printData<<std::endl;
                    if(!(exStatus.ok()))
                        std::cout<<"error!";    
                    // std::cout<<"computation is done"<<std::endl;
                    sendResult(outputs);
                    currentPipelineIdentifiers.clear();
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
