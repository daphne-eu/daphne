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

# Extending the DAPHNE Distributed Runtime

This page is about implementing and extending the distributed runtime. It focuses on
helping the DAPHNE developer understand the building blocks of the distributed runtime,
both of the DAPHNE coordinator as well as the DAPHNE distributed worker. If you want to
learn how to execute DaphneDSL scripts on the distributed runtime, please see related page in the [user documentation](/doc/DistributedRuntime.md).

## Background

The distributed runtime works similar to the vectorized engine. Compiler passes create
_pipelines_ by merging multiple operations. This allows DAPHNE to split data across multiple
workers (nodes) and let them execute a pipeline in parallel. The DAPHNE distributed runtime code
can be found at [`src/runtime/distributed`](/src/runtime/distributed) where it is split into two main parts. The
**coordinator** and the **worker**.

The DAPHNE distributed runtime works on an hierarchical approach.

1. Workers are **not** aware of the total execution. They handle a single task-pipeline and return outputs,
one pipeline at a time.
1. Workers are **only** aware of their chunk of data.

## Coordinator

The coordinator is responsible for distributing data and broadcasting the IR code fragment,
that is the task-pipeline. Each worker receives and compiles the IR optimizing it for its
local hardware. Coordinator code can be found at `src/runtime/distributed/coordinator/kernels`.

## Distributed Wrapper

The **`DistributedWrapper.h`** kernel contains the `DistributedWrapper` class which is the main entry point
for a distributed pipeline on the coordinator:

```cpp
void DistributedWrapper::execute(
                const char *mlirCode,       // the IR code fragment
                DT ***res,                  // array of pipeline outputs
                const Structure **inputs,   // array of pipeline inputs
                size_t numInputs,           // number of inputs
                size_t numOutputs,          // number of outputs
                int64_t *outRows,           // number of rows for each pipeline output
                int64_t *outCols,           // number of columns for each pipeline output
                VectorSplit *splits,        // compiler hints on how each input should be split
                VectorCombine *combines)    // compiler hints on how each output should be combined
```

Using the hints `splits` provided by the compiler we determine whether an input should be
**distributed/scattered** (by rows or columns) or **broadcasted**. Depending on that, we call
the corresponding kernel (`Distribute.h` or `Broadcast.h`, more details below) which is then
responsible for transferring data to the workers. The `DistributedCompute.h` kernel is then called
in order to broadcast the IR code fragment and start the actual computation.
Finally, the `DistributedCollect.h` kernel collects the final results (pipeline outputs).

Similar to local runtime kernels, **`Distribute.h`**, **`Broadcast.h`**, **`DistributedCompute.h`** and **`DistributedCollect.h`**
use C++ template meta programming (see the guidelines on [implementing built-in kernels](/doc/development/ImplementBuiltinKernel.md) for more information on how we utilize C++ templates on the local runtime). Since DAPHNE supports multiple distributed
backends (e.g., gRPC, MPI, etc.) we cannot provide a fully generic code that would work for
all implementations. Thus, we specialize each template for each distributed backend we want to
support.

The DAPHNE developer can work on a new distributed backend by simply providing a new template
specialization of these four kernels with a different distributed backend.

## Distributed Backends

DAPHNE supports multiple different devices (GPUs, distributed nodes, FPGAs etc.). Because of that,
all DAPHNE data objects (e.g. matrices) require tracking where their data is stored. That is why each
DAPHNE data object contains meta data providing that information. Below you can see a list of all the different
devices a DAPHNE data object can be allocated to.

Please note that not all of them are supported yet and we might add more in the future.

```cpp
enum class ALLOCATION_TYPE {
    DIST_GRPC,
    DIST_OPENMPI,
    DIST_SPARK,
    GPU_CUDA,
    GPU_HIP,
    HOST,
    HOST_PINNED_CUDA,
    FPGA_INT, // Intel
    FPGA_XLX, // Xilinx
    ONEAPI, // probably need separate ones for CPU/GPU/FPGA
    NUM_ALLOC_TYPES
};
```

As we described above, each kernel is partially specialized for each distributed backend. We specialize
the templated distributed kernels for each distributed allocation type in the `enum class` shown above
(e.g. `DIST_GRPC`).

## Templated Distributed Kernels

Each kernel is responsible for two things.

1. Handling the communication part. That is sending data to the workers.
2. Updating the meta data. That is populating the meta data of an object which can then lead
to reduced communication (if an object is already placed on nodes, we don't need to re-send it).

<!-- TODO: Add link to documentation of meta data. -->
You can find more information on the meta data implementation in `/src/runtime/local/datastructures`.

Below we see the **`Broadcast.h`** templated kernel, along with its gRPC specialization.
`DT` is the type of the object being broadcasted.

```cpp

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

// DT is object type
// AT is allocation type (distributed backend)
template<ALLOCATION_TYPE AT, class DT>
struct Broadcast {
    static void apply(DT *mat, bool isScalar, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<ALLOCATION_TYPE AT, class DT>
void broadcast(DT *&mat, bool isScalar, DCTX(dctx))
{
    Broadcast<AT, DT>::apply(mat, isScalar, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// GRPC
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_GRPC, DT>
{
    // Specific gRPC implementation...
    ...
```

So, for example, if we wanted to add broadcast support for MPI, all we need to do is provide the partial
template specialization of the `Broadcast` kernel for MPI.

```cpp
// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------

template<class DT>
struct Broadcast<ALLOCATION_TYPE::DIST_MPI, DT> 
{
    // Specific MPI implementation...
    ...
```

For now, the selection of a distributed backend is hardcoded in `/src/runtime/distributed/coordinator/kernels/DistributedWrapper.h`.
<!-- 
TODO: PR #436 provides support for MPI and implements a cli argument for selecting a distributed backend. This section will be updated once #436 is merged.
 -->

## Distributed Worker

The worker code can be found in `src/runtime/distributed/worker`.

**`WorkerImpl.h`** contains the `WorkerImpl` class which provides all the logic for the distributed worker.
There are three important methods in this class:

- The **`Store`** method, which stores an object in memory and returns an identifier.
- The **`Compute`** method, which receives the IR code fragment along with identifier of inputs, computes the pipeline and returns identifiers of pipeline outputs.
- And the **`Transfer`** method, which is used to return an object using an identifier.

```cpp
/**
 * @brief Stores a matrix at worker's memory
 * 
 * @param mat Structure * obj to store
 * @return StoredInfo Information regarding stored object (identifier, numRows, numCols)
 */
StoredInfo Store(DT *mat) ;

/**
 * @brief Computes a pipeline
 * 
 * @param outputs vector to populate with results of the pipeline (identifier, numRows/cols, etc.)
 * @param inputs vector with inputs of pipeline (identifiers to use, etc.)
 * @param mlirCode mlir code fragment
 * @return WorkerImpl::Status tells us if everything went fine, with an optional error message
 */
Status Compute(vector<StoredInfo> *outputs, vector<StoredInfo> inputs, string mlirCode) ;

/**
 * @brief Returns a matrix stored in worker's memory
 * 
 * @param storedInfo Information regarding stored object (identifier, numRows, numCols)
 * @return Structure* Returns object
 */
Structure * Transfer(StoredInfo storedInfo);
```

The developer can provide an implementation for a distributed worker by deriving from the `WorkerImpl` class.
The derived class handles all the communication using the preferred distributed backend and invokes the parent methods for the logic.
You can find the gRPC implementation of the distributed worker in `src/runtime/distributed/worker/WorkerImplGrpc.h/.cpp`.

`main.cpp` is the entry point of the distributed worker. A distributed implementation is created using a pointer to the parent class
`WorkerImpl`. The distributed node then blocks and waits for the coordinator to send a request by invoking the virtual method:

```cpp
virtual void Wait() { };
```

Each distributed worker implementation needs to override this method and implement it in order to wait for requests.
