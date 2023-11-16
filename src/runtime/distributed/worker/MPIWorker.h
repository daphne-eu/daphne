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

#include <runtime/distributed/worker/WorkerImpl.h>
#include <runtime/distributed/worker/MPIHelper.h>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <runtime/local/io/DaphneSerializer.h>

class MPIWorker : WorkerImpl {
    public:
        MPIWorker(DaphneUserConfig& _cfg) : WorkerImpl(_cfg) {//TODO
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
        // Store in chunks
        std::unique_ptr<DaphneDeserializerChunks<Structure>> deserializer;
        std::unique_ptr<DaphneDeserializerChunks<Structure>::Iterator> deserializerIter;
        Structure *deserializedMatrix; 

        std::tuple<bool, StoredInfo> storeInputs(const std::vector<char> &buffer, size_t messageLength)
        {
            StoredInfo info;
            if (*deserializerIter == deserializer->begin() && DF_Dtype(buffer.data()) == DF_data_t::Value_t) {
                double val = DaphneSerializer<double>::deserialize(buffer.data());
                info = this->Store(&val);
                return std::make_tuple(true, info);
            } else {
                // partially deserialize next
                (*deserializerIter)->first = messageLength;
                if ((*deserializerIter)->second->size() < messageLength)
                    (*deserializerIter)->second->resize(messageLength);
                (*deserializerIter)->second->assign(static_cast<const char*>(buffer.data()), static_cast<const char*>(buffer.data()) + messageLength);
                
                // advance iterator, this also partially deserializes
                ++(*deserializerIter);
                // response if we completed
                if (*deserializerIter == deserializer->end()){
                    info = this->Store(deserializedMatrix);
                    return std::make_tuple(true, info);
                }
            }            
            return std::make_tuple(false, info); 
        }
        
        void sendComputeResult(std::vector<StoredInfo> outputs)
        {
            std::string computeResult = ""; 
            StoredInfo tempInfo=outputs.at(0);
            computeResult += tempInfo.toString();
            for (size_t i = 1; i < outputs.size(); i++)
            {   
                StoredInfo tempInfo=outputs.at(i);
                computeResult += ":" + tempInfo.toString();
            }
            MPI_Send(computeResult.c_str(), computeResult.size(), MPI_CHAR, COORDINATOR, COMPUTERESULT, MPI_COMM_WORLD);
        }
        void sendMatrix(StoredInfo info)
        {
            auto mat = this->Transfer(info);
            std::vector<char> dataToSend;
            size_t messageLength = DaphneSerializer<Structure>::serialize(mat, dataToSend);
            
            int len = messageLength;
            MPI_Send(dataToSend.data(), len, MPI_UNSIGNED_CHAR, COORDINATOR, OUTPUT, MPI_COMM_WORLD);            
        }
        
        void prepareBufferForMessage(std::vector<char> &buffer, int * messageLength, MPI_Datatype type, int source, int tag)
        {
            MPI_Status messageStatus;
            MPI_Recv(messageLength, 1, type, source, tag, MPI_COMM_WORLD, &messageStatus);
            // std::cout<< id<<" in distribute size " <<*messageLength << " tag " << tag <<std::endl;            
            if (buffer.size() < size_t(*messageLength))
                buffer.resize(size_t(*messageLength));
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
            std::vector<char> buffer;
            int messageLength;
            MPIHelper::Task MsgTask;
            std::string printData="";
            std::vector<StoredInfo> outputs;
            std::string identifier;
            WorkerImpl::Status exStatus(true);
            switch(tag){
                case STREAM_INIT:
                    int chunkSize;
                    MPI_Recv(&chunkSize, 1, MPI_INT, COORDINATOR, STREAM_INIT, MPI_COMM_WORLD, &messageStatus);
                    deserializer.reset(new DaphneDeserializerChunks<Structure>(&deserializedMatrix, chunkSize));
                    deserializerIter.reset(new DaphneDeserializerChunks<Structure>::Iterator(deserializer->begin()));    
                    (*deserializerIter)->second->resize(chunkSize);
                break;
                case DATASIZE: {
                    prepareBufferForMessage(buffer, &messageLength, MPI_INT, source, DATASIZE);
                    MPI_Recv(buffer.data(), messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, DATA, MPI_COMM_WORLD, &messageStatus);                    
                    auto ret = storeInputs(buffer, (size_t)messageLength);
                    if (std::get<0>(ret))
                        sendDataACK(std::get<1>(ret));
                }
                break;
                case BROADCAST: {
                    prepareBufferForMessage(buffer, &messageLength, MPI_INT, source, BROADCAST);
                    MPI_Bcast(buffer.data(), messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    auto ret = storeInputs(buffer, (size_t)messageLength);
                    if (std::get<0>(ret))
                        sendDataACK(std::get<1>(ret));
                }                
                break;   

                case MLIRSIZE:
                    prepareBufferForMessage(buffer, &messageLength, MPI_INT, source, MLIRSIZE);
                    MPI_Recv(buffer.data(), messageLength, MPI_UNSIGNED_CHAR, COORDINATOR, MLIR,MPI_COMM_WORLD, &messageStatus);
                    MsgTask.deserialize(buffer);
                    
                    exStatus = this->Compute(&outputs, MsgTask.inputs, MsgTask.mlir_code);
                    sendComputeResult(outputs);                    
                break;

                case TRANSFERSIZE: {
                    prepareBufferForMessage(buffer, &messageLength, MPI_INT, source, TRANSFERSIZE);
                    MPI_Recv(buffer.data(), messageLength, MPI_CHAR, COORDINATOR, TRANSFER, MPI_COMM_WORLD, &messageStatus);
                    auto info = MPIHelper::constructStoredInfo(std::string(buffer.data()));
                    sendMatrix(info);
                }
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
