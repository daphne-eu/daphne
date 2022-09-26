<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");>
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running DAPHNE on the distributed runtime

## Background

Daphne supports execution in a distributed fashion. Similar to the local vectorized engine, the distributed runtime
uses multiple distributed nodes (workers) that work on their local data, while a main node, the coordinator, is responsible
for transferring the data and code to be executed. The user is required to start the workers, either manually or using an 
HPC tool as SLURM (scripts that start the workers locally or remotely, natively or not, can be found [here](https://github.com/daphne-eu/daphne/tree/main/deploy)). 
<!-- TODO: add link to documentation. -->

##  Scope

This document focuses on:
- how to start distributed workers
- executing Daphne scripts on the distributed runtime


## Build the daphne prototype

First you need to build the Daphne prototype. This doc assumes that you already built Daphne and can run it locally. If 
you need help building or running Daphne see [here](https://github.com/daphne-eu/daphne/blob/main/doc/GettingStarted.md).

## Start distributed workers

Before executing Daphne on the distributed runtime, worker nodes must first be up and running. You can start a distributed worker within the Daphne prototype directory as follows:

```bash
# IP:PORT is the IP and PORT the worker server will be listening too
./build/src/runtime/distributed/worker/DistributedWorker IP:PORT 
```

There are [scripts](https://github.com/daphne-eu/daphne/tree/main/deploy) that automate this task and can help running multiple workers at once 
locally or even utilizing tools (like SLURM) in HPC environments.

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
./build/bin/daphne --distributed ./example.script
```

### What Next?

You might want to have a look at
- the [distributed runtime development guideline](/doc/development/ExtendingDistributedRuntime.md)
- the [contribution guidelines](/CONTRIBUTING.md)
- the [open distributed related issues](https://github.com/daphne-eu/daphne/issues?q=is%3Aopen+is%3Aissue+label%3ADistributed)