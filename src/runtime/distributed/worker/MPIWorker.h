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
#include <unistd.h>
#include  <iostream>

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST=0, DISTRIBUTE, DETACH, DATA, MLIR
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};
class MPIWorker{
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
                    handleInCommingMessages(status);
                else
                    continueComputing(); // takes form a queue // hocks for scheuling
            }
        }
    private:
        int id;
        int myState=LISTENING;
        int temp=0;
        void detachFromComputingTeam(){
            myState = DETACHED;
            std::cout<<"I am " << id <<". I got detach message... " << std::endl;
        }
        void terminate(){
            myState = TERMINATED;
            std::cout<<"I am worker " << id << ". I'll rest in peace" << std::endl;
        }
        void continueComputing(){
            if(temp<3){ //mimic still having computations 
                sleep(1);
                temp++;
            }
            else if(myState == DETACHED){ 
                terminate();
            }
        }
        void handleInCommingMessages(MPI_Status status){
            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;
            int size;
            MPI_Get_count(&status, MPI_CHAR, &size);
            int dataSize;
            int codeSize; 
            MPI_Status messageStatus;
            unsigned char  * info;
            unsigned char * data;
            char * mlirCode;
            switch(tag){
                case BROADCAST:
                    /*info = (unsigned char *) malloc(size * sizeof(unsigned char));
                    MPI_Recv(info, size, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &messageStatus);
                    MPISerializer::getSizeOfMessage(info, &dataSize);
                    data = (unsigned char *) malloc (dataSize * sizeof(unsigned char));
                    MPI_Bcast(data, dataSize, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
                    MPISerializer::deserialize(data);
                    free(info);
                    free(data);*/
                break;

                case DISTRIBUTE:
                    // TODO
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
