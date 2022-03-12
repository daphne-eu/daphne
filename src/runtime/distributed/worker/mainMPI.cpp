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

#include "MPIProcess.h"

int main (int argc, char ** argv) {
    MPIProcess mpiProc(argc, argv);
    

    //init 2d array
    int ** arr;
    int row = 4; int col = 3;
    arr = new int * [row]; //allocate rows
    for(int i=0; i< row; i++){
        arr[i] = new int[col]; //allocate cols
    }

    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            arr[i][j] = 5;
        }
    }

   // mpiProc.runMPI(1, 1, 2, arr, row, col);
   // mpiProc.runMPI(2, 1, 2, arr, row, col);
   // mpiProc.runMPI(3, 0, 2, arr, row, col);
   // mpiProc.runMPI(4, 0, 2, arr, row, col);
    mpiProc.runMPI(5, NULL, 0, NULL, row, col);
    
    mpiProc.freeMatrix(arr);
}