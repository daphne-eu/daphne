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

## Background

DAPHNE supports execution in a distributed fashion. Utilizing the DAPHNE distributed runtime does not require any changes to the DaphneDSL script.
Similar to the local vectorized engine ([here, section 4](https://daphne-eu.eu/wp-content/uploads/2022/08/D2.2-Refined-System-Architecture.pdf)), the compiler automatically fuses operations and creates pipelines for the distributed runtime, which then uses multiple distributed nodes (workers) that work on their local data, while a main node, the coordinator, is responsible for transferring the data and code to be executed.
As mentioned above, changes to the DaphneDSL code are not needed, however the user is required to start the workers, either manually or using an HPC tool such as Slurm (see the documentation on [deployment](/deploy) for scripts that start the workers locally or remotely, natively or not).

## Scope

This document focuses on:

- how to start distributed workers
- executing DaphneDSL scripts on the distributed runtime
- DAPHNE's distributed runtime has two different backends. This page explains how things work with the **gRPC backend**.
A brief introduction to the other backend using **OpenMPI** can be viewed in [this document](MPI-Usage.md).

## Build DAPHNE

First you need to build DAPHNE. This document assumes that you have already built DAPHNE and can run it locally. If you need help building or running DAPHNE see the guidelines for [getting started](/doc/GettingStarted.md).

## Building the Distributed Worker

The DAPHNE distributed worker is a different executable which can be built using the build script by providing the `--target` argument:

```bash
./build.sh --target DistributedWorker
```

## Start Distributed Workers

Before executing DAPHNE on the distributed runtime, worker nodes must first be up and running. You can start a distributed worker within the DAPHNE directory as follows:

```bash
# IP:PORT is the IP and PORT the worker server will be listening too
bin/DistributedWorker IP:PORT 
```

There are [scripts](/deploy) that automate this task and can help running multiple workers at once locally or even utilizing tools (like Slurm) in HPC environments.

Each worker can be left running and reused for multiple scripts and pipeline executions (however, for now they might run into memory issues, see **Limitations** section below).

Each worker can be terminated by sending a `SIGINT` (Ctrl+C) or by using the scripts mentioned above.

## Set up Environment Variables

After setting up the workers, before we run DAPHNE we need to specify which IPs
and ports the workers listen to. For now, we use an environment variable called
`DISTRIBUTED_WORKERS` where we list IPs and ports of the workers separated by comma.

```bash
# Example for 2 workers.
# Worker1 listens to localhost:5000
# Worker2 listens to localhost:5001
export DISTRIBUTED_WORKERS=localhost:5000,localhost:5001
```

## Run DAPHNE Using the Distributed Runtime

Now that we have all workers up and running and the environment variable is set, we can run DAPHNE. We enable the distributed runtime by specifying the flag `--distributed`.

**(*)** Note that we execute DAPHNE from the same bash shell we've set up the environment variable  `DISTRIBUTED_WORKERS`.

```bash
bin/daphne --distributed ./example.script
```

Currrently, synchronous gRPC, asynchronous gRPC, and [OpenMPI](/doc/MPI-Usage.md) are implemented as distributed backends.
The distributed backend can be selected using the command-line argument `--dist_backend`.

## Example

On one terminal we start up a distributed worker:

```bash
$ bin/DistributedWorker localhost:5000
Started Distributed Worker on `localhost:5000`
```

On another terminal we set the environment variable and execute the script [`distributed.daph`](/scripts/examples/distributed.daph):

```bash
$ export DISTRIBUTED_WORKERS=localhost:5000
$ bin/daphne --distributed ./scripts/example/distributed.daph
```

## Current Limitations

The distributed runtime is still under development and currently there are various limitations. Most of these limitations will be fixed in the future.

- Distributed runtime for now heavily depends on the vectorized engine and how pipelines are
created and multiple operations are fused together (more [here - section 4](https://daphne-eu.eu/wp-content/uploads/2022/08/D2.2-Refined-System-Architecture.pdf)). This causes some limitations related to pipeline creation (e.g. [not supporting pipelines with different result outputs](https://github.com/daphne-eu/daphne/tree/main/issues/397) or pipelines with no outputs).
- For now, the distributed runtime only supports the `DenseMatrix` data type and the `double` value type, i.e., `DenseMatrix<double>` (issue [#194](https://github.com/daphne-eu/daphne/tree/main/issues/194)).
- A DAPHNE pipeline input might exist multiple times in the input array. For now, this is not supported. In the future, similar pipelines will simply omit multiple pipeline inputs and each one will be provided only once.
- Garbage collection at worker (node) level is not implemented yet. This means that after some time the workers can fill up their memory completely, requiring a restart.