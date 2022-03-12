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

#include "mpi.h"
#include <iostream>
#include <stdlib.h>
#define N 8
using namespace std;
class MPIProcess{
    public:
        MPIProcess(int & argc, char** & argv);
        ~MPIProcess();
        int getRank() const;
        int getNumberOfProcesses() const;
        int getRank(const MPI_Comm & communicator) const;
        int getNumberOfProcesses(const MPI_Comm & communicator) const;
        void freeMatrix(int ** mat);
        char * getHostName() const;
        void fill(int n, int m, double ** matrix, int nr);
        //MPI_functionality
        int runMPI(int option, int senderRank, int recvRank, int** arr, int row, int col);
        
    private:
        int myRank;
        int numberOfProcesses;
        char * hostname; 
};

const int HOSTNAME_LENGTH = 64;

inline MPIProcess::MPIProcess(int &argc, char** &argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    hostname = new char[HOSTNAME_LENGTH];
    int nameLength;
    MPI_Get_processor_name(hostname, &nameLength);
    cout << myRank << " on " <<  hostname << " is initialized!" <<endl;
}

inline MPIProcess::~MPIProcess(){
    MPI_Finalize();
    delete [] hostname;
}

inline void MPIProcess::fill(int n, int m, double ** matrix, int nr){
    double * data = (double *) malloc(sizeof(double) * n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i*N + j] = i*N + j;
        }
    }
    *matrix = data;
}

inline char * MPIProcess::getHostName() const{
    return hostname;
}

inline int MPIProcess::getRank() const{
    return myRank;
}

inline int MPIProcess::getNumberOfProcesses() const{
    return numberOfProcesses;
}

inline int MPIProcess::getRank(const MPI_Comm & communicator) const {
    int commRank;
    MPI_Comm_rank(communicator, &commRank);
    return commRank;
}

inline int MPIProcess::getNumberOfProcesses(const MPI_Comm & communicator) const {
    int commProcesses;
    MPI_Comm_size(communicator, &commProcesses);
    return commProcesses;
}

inline int MPIProcess::runMPI(int option, int senderRank, int recvRank, int** arr, int row, int col){
    int rowChunkSize;

    if(option != 3 && option !=4 && option !=5){
        if(myRank != senderRank && myRank != recvRank) {
            option = 999;
        }
    }

    cout<<"running option "<<option<<endl;
    switch(option){
        case 1:
        {
            cout << "do MPI send rank "<<myRank<<"\n";
            arr[0][0] = 3;
            if(myRank == senderRank){
                MPI_Send(&(arr[0][0]), row*col, MPI_INT, recvRank, 0, MPI_COMM_WORLD);
            }
            break;
        }
        case 2:
        {
            cout << "do MPI recv "<<myRank<<"\n";
            MPI_Status status;
            if(myRank == recvRank){
                MPI_Recv(&(arr[0][0]), row*col, MPI_INT, senderRank, 0, MPI_COMM_WORLD, &status);
                cout<<"received array content"<<endl;
                for(int i=0; i<row; i++){
                    for(int j=0; j<col; j++){
                        cout<<arr[i][j]<<" ";
                    }
                    cout<<endl;
                }
            }
            break;
        }
        case 3:
        {
            
            cout << "do MPI broadcast \n";
            if(myRank==senderRank){
                arr[0][0] = 4;
                
            }
            MPI_Bcast(&(arr[0][0]), row*col, MPI_INT, senderRank, MPI_COMM_WORLD);
            cout<<"received array content from broadcast at rank"<<myRank<<endl;
            for(int i=0; i<row; i++){
                for(int j=0; j<col; j++){
                    cout<<arr[i][j]<<" ";
                }
                cout<<endl;
            }
            
            break;
        }
        case 4:
        {
            // cout<<"do MPI Scatter at "<<myRank<<endl;
            rowChunkSize = row / numberOfProcesses;
            int temp[rowChunkSize][col];
            //e.g split by row
            
            cout<<"bleh"<<endl;
            for(int i=0; i<rowChunkSize; i++){
                for(int j=0; j<col; j++){
                    cout<<"before received data at rank "<<myRank<<":"<<temp[i][j]<<" ";
                }
                cout<<endl;
            }

            //convert 2d into 1d array
            int * data = (int *) malloc(sizeof(int) * row * col);
            for (int q = 0; q < row; q++)
            {
                for (int t = 0; t < col; t++)
                {
                    data[q * col + t] = arr[q][t];
                }
            }

            MPI_Scatter(data, row*col/numberOfProcesses, MPI_INT, temp, row*col/numberOfProcesses, MPI_INT, senderRank, MPI_COMM_WORLD);
            
        
            
            for(int i=0; i<rowChunkSize; i++){
                for(int j=0; j<col; j++){
                    cout<<"received data at rank "<<myRank<<":"<<temp[i][j]<<" ";
                }
                cout<<endl;
            }
            // int rows;  
            // double *matrix_A = NULL;
            // rows = N / numberOfProcesses;

            // if(myRank == 0){                          
            //     fill(N, N, &matrix_A, 10);   
            // }

            // double cc[rows][N];

            // MPI_Scatter(matrix_A, N*N/numberOfProcesses, MPI_DOUBLE, cc, N*N/numberOfProcesses, MPI_DOUBLE, 0, MPI_COMM_WORLD);        

            // for (int i = 0; i < rows; i++) {
            //     for (int j = 0; j < N; j++) {
            //         cout<<myRank<<": "<<cc[i][j]<<"  ";
            //     }
            //     cout<<endl;
            // }
            delete[] data;
            break;
        }
        case 5:
        {
            
            //do MPI gather on row basis
            rowChunkSize = row / numberOfProcesses;
            // int ** arrglobal; int ** arrlocal;
            //arrglobal = new int * [row]; //allocate rows
            // for(int i=0; i< row; i++){
            //     arrglobal[i] = new int[col]; //allocate cols
            // }
            
            int arrglobal[row][col];
            // const int c_col = col;
            // int arrglobal  = new int[row][c_col];
            

            //arrlocal = new int * [rowChunkSize]; //allocate rows
            int arrlocal[rowChunkSize][col];
            //int arrlocal = new int[row][c_col];
            // for(int i=0; i< rowChunkSize; i++){
            //     arrlocal[i] = new int[col]; //allocate cols
            // }

            for(int i=0; i < rowChunkSize; i++){
                for(int j=0; j < col; j++){
                    arrlocal[i][j] = 5+myRank;
                }
            }
            
            

            MPI_Datatype subRows;
            MPI_Type_vector(rowChunkSize, col, col, MPI_INT, &subRows);
            MPI_Type_commit(&subRows);

            cout<<"do MPI Gather"<<endl;
            MPI_Gather(&(arrlocal[0][0]),  rowChunkSize*col, MPI_INT, &(arrglobal[0][0]), 1, subRows, recvRank, MPI_COMM_WORLD);
            if(myRank == recvRank){
                for (int i = 0; i < row; i++) {
                    for (int j = 0; j < col; j++) {
                        cout<<myRank<<":"<<arrglobal[i][j]<<"  ";
                    }
                    cout<<endl;
                }
            }
            

            //freeMatrix(arrlocal);
            //freeMatrix(arrglobal);
            // for (int i = 0; i < row; i++) {
            //     delete [] arrglobal[i];
            // }
            // delete [] arrglobal;
            // arrglobal = 0;
            // for (int i = 0; i < rowChunkSize; i++) {
            //     delete [] arrlocal[i];
            // }
            // delete [] arrlocal;
            //arrlocal = 0;
            MPI_Type_free(&subRows);


            //need to handle rows partition if the module with nProc is not equal to 0
            //lets try without mpi_type_vector, such as converting the matrix into 1d array shape
            break;
        }
        default:
            cout<<"rank "<<myRank<<" is doing nothing"<<endl;
            return 0;
    }
    return 1;
}

inline void MPIProcess::freeMatrix(int** matrix){
    free(matrix[0]);

    free(matrix);
}



