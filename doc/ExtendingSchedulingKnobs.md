<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Extending DAPHNE with more scheduling knobs

This document focuses on:
- how a daphne end-user may extend the DAPHNE system by adding new scheduling techniques

### Guidelines

The end-user should consider the following files for adding a new scheduling technique
1. src/runtime/local/vectorized/LoadPartitioning.h
2. src/api/cli/daphne.cpp

**Adding the actual code of the technique**

The first file `LoadPartitioning.h` contains the implementation of the currently supported scheduling techniques, i.e., the current version of DAPHNE uses self-scheduling techniques to partition the tasks. Also, it uses the self-scheduling principle for executing the tasks. 
For more details, please visit [Scheduler design for tasks and pipelines](https://daphne-eu.eu/wp-content/uploads/2021/11/Deliverable-5.1-fin.pdf).

In this file, the user should change two things:
1. The enumeration that is called `SelfSchedulingScheme`. The user will have to add a name for the new technique, e.g., `MYTECH`

```c++
enum SelfSchedulingScheme { STATIC=0, SS, GSS, TSS, FAC2, TFSS, FISS, VISS, PLS, MSTATIC, MFSC, PSS, MYTECH };
```

2. The function that is called `getNextChunk()`. This function has a switch case that selects the mathematical formula that corresponds to the chosen scheduling method. The user has to add a new case to handle the new technique.

```c++
 uint64_t getNextChunk(){
    //...
    switch (schedulingMethod){
        //...
        //Only the following part is what the user has to add. The rest remains the same
        case MYTECH:{ // the new technique
            chunkSize= FORMULA;//Some Formula to calculate the chunksize (partition size)
            break; 
        }
        //...
    }
    //...
    return chunkSize;
 }
            
``` 
**Enabling the selection of the newly added technique**

The second file `daphne.cpp` contains the code that parses the command line arguments and passes them to the DAPHNE compiler and runtime. The user has to add the new technique as a vaild option. Otherwise, the users will not be able to use the newly added technique. 
There is a variable called `taskPartitioningScheme` and it is of type `opt<SelfSchedulingScheme>`.
The user should extend the declaration of `opt<SelfSchedulingScheme>` as follows:
```c++
opt<SelfSchedulingScheme> taskPartitioningScheme(
        cat(daphneOptions), desc("Choose task partitioning scheme:"),
        values(
            clEnumVal(STATIC , "Static (default)"),
            clEnumVal(SS, "Self-scheduling"),
            clEnumVal(GSS, "Guided self-scheduling"),
            clEnumVal(TSS, "Trapezoid self-scheduling"),
            clEnumVal(FAC2, "Factoring self-scheduling"),
            clEnumVal(TFSS, "Trapezoid Factoring self-scheduling"),
            clEnumVal(FISS, "Fixed-increase self-scheduling"),
            clEnumVal(VISS, "Variable-increase self-scheduling"),
            clEnumVal(PLS, "Performance loop-based self-scheduling"),
            clEnumVal(MSTATIC, "Modified version of Static, i.e., instead of n/p, it uses n/(4*p) where n is number of tasks and p is number of threads"),
            clEnumVal(MFSC, "Modified version of fixed size chunk self-scheduling, i.e., MFSC does not require profiling information as FSC"),
            clEnumVal(PSS, "Probabilistic self-scheduling"),
            clEnumVal(MYTECH, "some meaningful description to the abbrevaition of the new technique")
        )
); 
```

**Usage of the new technique**

Users may now pass the new technique as an option when they execute a DaphneDSL script.
```bash
daphne --vec --MYTECH --grain-size 10 --num-threads 4 my_script.daphne
```
In this example, the daphne system will execute `my_script.daphne`  with the following configuration:
1. the vectorized engine is enabled due to `--vec`
2. the DAPHNE runtime will use MYTECH for task partitioning due to `--MYTECH`
3. the minimum partition size will be 10 due to `--grain-size 10 ` 
4. the vectorized engine will use 4 threads due to `--num-threads 4` 
