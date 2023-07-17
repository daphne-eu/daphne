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

# Distributed Runtime

Running DAPHNE on the Distributed Runtime

## Background

Daphne supports execution in a distributed fashion. Utilizing the Daphne Distributed Runtime does not require any changes to the DaphneDSL script.
Similar to the local vectorized engine ([here, section 4](https://daphne-eu.eu/wp-content/uploads/2022/08/D2.2-Refined-System-Architecture.pdf)), the compiler automatically fuses operations and creates pipelines for the distributed runtime, which then uses multiple distributed nodes (workers) that work on their local data, while a main node, the coordinator, is responsible for transferring the data and code to be executed.
As mentioned above, changes at DaphneDSL code are not needed, however the user is required to start the workers, either manually or using an HPC tool as SLURM (scripts that start the workers locally or remotely, natively or not, can be found [here](/deploy)).
<!-- TODO: add link to documentation. -->

## Scope

This document focuses on:

- how to start distributed workers
- executing Daphne scripts on the distributed runtime
- DAPHNE's distributed runtime has two different backends. This page explains how things work with the **gRPC backend**.
A brief introduction to the other backend using **OpenMPI** can be viewed in [this document](MPI-Usage.md).

## Build the Daphne prototype

First you need to build the Daphne prototype. This doc assumes that you already built Daphne and can run it locally. If you need help building or running Daphne see [here](/doc/GettingStarted.md).

## Building the Distributed Worker

The Daphne distributed worker is a different executable which can be build using the build-script and providing the `--target` argument:

```bash
./build.sh --target DistributedWorker
```

## Start distributed workers

Before executing Daphne on the distributed runtime, worker nodes must first be up and running. You can start a distributed worker within the Daphne prototype directory as follows:

```bash
# IP:PORT is the IP and PORT the worker server will be listening too
./bin/DistributedWorker IP:PORT 
```

There are [scripts](/deploy) that automate this task and can help running multiple workers at once locally or even utilizing tools (like SLURM) in HPC environments.

Each worker can be left running and reused for multiple scripts and pipeline executions (however, for now they might run into memory issues, see **Limitations** section below).

Each worker can be terminated by sending a `SIGINT` (Ctrl+C) or by using the scripts mentioned above.

## Set up environmental variables

After setting up the workers, before we run Daphne we need to specify which IPs
and ports the workers are listening too. For now we use an environmental variable called
`DISTRIBUTED_WORKERS` where we list IPs and ports of the workers separated by a comma.

```bash
# Example for 2 workers.
# Worker1 listens to localhost:5000
# Worker2 listens to localhost:5001
export DISTRIBUTED_WORKERS=localhost:5000,localhost:5001
```

## Run DAPHNE using the distributed runtime

Now that we have all workers up and running and the environmental variable is set we can run Daphne. We enable the distributed runtime by specifying the flag argument `--distributed`.

**(*)** Note that we execute Daphne from the same bash shell we've set up the environmental variable  `DISTRIBUTED_WORKERS`.

```bash
./bin/daphne --distributed ./example.script
```

For now only asynchronous-gRPC is implemented as a distributed backend and selection is hardcoded [here](/src/runtime/distributed/coordinator/kernels/DistributedWrapper.h#L73).
<!-- 
TODO: PR #436 provides support for MPI and implements a cli argument for selecting a distributed backend. This section will be updated once #436 is merged.
 -->

## Example

On one terminal with start up a Distributed Worker:

```bash
$./bin/DistributedWorker localhost:5000
Started Distributed Worker on `localhost:5000`
```

On another terminal we set the environment variable and execute script [`distributed.daph`](/scripts/examples/distributed.daph):

```bash
$ export DISTRIBUTED_WORKERS=localhost:5000
$ ./bin/daphne --distributed ./scripts/example/distributed.daph
```

## Current limitations

Distributed Runtime is still under development and currently there are various limitations. Most of these limitations will be fixed in the future.

- Distributed runtime for now heavily depends on the vectorized engine of Daphne and how pipelines are
created and multiple operations are fused together (more [here - section 4](https://daphne-eu.eu/wp-content/uploads/2022/08/D2.2-Refined-System-Architecture.pdf)). This causes some limitations related to pipeline creation (e.g. [not supporting pipelines with different result outputs](/issues/397) or pipelines with no outputs).
- For now distributed runtime only supports `DenseMatrix` types and value types `double` - `DenseMatrix<double>` (issue [#194](/issues/194)).
- A Daphne pipeline input might exist multiple times in the input array. For now this is not supported. In the future similar pipelines will simply omit multiple pipeline inputs and each one will be provided only once.
- Garbage collection at worker (node) level is not implemented yet. This means that after some time the workers can fill up their memory completely, requiring a restart.

## What Next?

You might want to have a look at

- the [distributed runtime development guideline](/doc/development/ExtendingDistributedRuntime.md)
- the [contribution guidelines](/CONTRIBUTING.md)
- the [open distributed related issues](https://github.com/daphne-eu/daphne/issues?q=is%3Aopen+is%3Aissue+label%3ADistributed)
