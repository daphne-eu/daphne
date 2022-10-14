<!--
Copyright 2022 The DAPHNE Consortium

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

# DAPHNE Deployment

### Where to Start

Use [deploy/deploy-distributed-on-slurm.sh](./deploy-distributed-on-slurm.sh) to start. When executed without parameters, it prints out the help message.

### Overview

This directory [deploy/](../deploy/) can be used to **deploy DAPHNE** Runtime Systems.
Deploying allows the [source code](../src/) to be:
- built as Target Runtime(s) using [build.sh](../build.sh),
- packaged,
- delivered and installed on to the Deployment Platform (e.g. HPC) as a unifying Deployment Platform of these Runtime Systems, and 
- run on the Resources of a set of distributed components.
It can also be used to just try out DAPHNE on a single Runtime System setup.

Once a deployment is running, the Distributed Workers (started `DistributedWorker` Target Runtime(s) executables) are running in their Runtime Systems and the main Target Runtimes use them to execute Daphne scripts in a Distributed Deployment.

### Computer Architecture Framework

DAPHNE Deployment encompases the following HPC Computer Architecture Framework:

- Deployment Platform (e.g. an HPC with SLURM support)
  - Runtime System(s)
    - Runtime System for the Main Target or
      - `daphne` Target
    - Runtime System for the Distributed Worker Target
      - `DistributedWorker` Target

```
                    DAPHNE Deployment
            a Computer Architecture Framework

+--------------------------------------------------------------------------------------+
|                                                                                      |
|   +------------------+                                                               |
|   | Compilation node |                                                               |
|   |                  |                                                               |
|   +------------------+                                                               |
|       |                                                                              |
|       |                                                                              |
|       | (SSH connection)                                                             |
|       |                                                                              |
|       |                                                                              |
| +----------------------------------------------------------------------------------+ |
| |Deployment Platform (e.g. an HPC with SLURM support)                              | |
| |                                                                                  | |
| |  +------------------------------+                                                | |
| |  | Access/Submission/Login Node |                                                | |
| |  |                              |                                                | |
| |  +------------------------------+                                                | |
| |      |                                                                           | |
| |      |                                                                           | |
| |      |   Network connections, e.g. Infiniband, to e.g. SLURM interfaces,         | |
| |      |   used also for communications between MT and DWs.                        | |
| |      |-------------------------------------------------------------------+       | |
| |      |                                         |                         |       | |
| |  +--------------------------+     +--------------------------+     +-----------+ | |
| |  | Node 1                   |     | Node 2                   |     | Node n    | | |
| |  | - Resources              | ... |                          | ... |           | | |
| |  |   - CPU/GPU/FPGA         |     | CPU/GPU/FPGAs            |     | Resources | | |
| |  | - Runtime Systems        |     |   (e.g. 128+)            |     |           | | |
| |  |   - Main Target (MT) 1   |     | {DistributedWorker (DW)} |     | DWs       | | |
| |  |   - (optional: more DWs) |     |   (e.g. DWs 1..128)      |     |           | | |
| |  +--------------------------+     +--------------------------+     +-----------+ | |
| |                                                                                  | |
| +----------------------------------------------------------------------------------+ |
|                                                                                      |
+--------------------------------------------------------------------------------------+
```

### Deployment Functionalities

This directory includes a set of **shell files** regarding deployment aspects like:
- packaging/virtualization of the deployment (installation) package,
- containerized packaging,
- virtualized installation,
- managed deployment,
- deployment of the ˙daphne˙ executable,
- distributed starting of the Target Runtimes within Runtime Systems, and
- collection and analysis of experimental data obtained through running of a deployment.

### List of Files in this Directory

The list of files in this directory with their description:

1. [deploy/README.md](README.md) (this file) 
  - A short README file to explain directory structure and point to more documentation in [doc/Deploy.md](../doc/Deploy.md).
2. [deploy/build-daphne-singularity-image.sh](build-daphne-singularity-image.sh)
  - This script builds the "daphne.sif" singularity image from the Docker image ahmedeleliemy/test-workflow, also contained in this folder.
3. [deploy/deploy-distributed-on-slurm.sh](deploy-distributed-on-slurm.sh)
  - This script allows the user to deploy DAPHNE through SLURM.
4. [deploy/deployDistributed.sh](deployDistributed.sh)
  - This script sends and builds executables on each node (basic version).
5. [deploy/Dockerfile](Dockerfile)
  - A `Dockerfile` based on `ubuntu:latest` with some other libraries. It is also used for building a Singularity image that is converted from this Docker image.
6. [deploy/example-time.daphne](example-time.daphne)
  - A Daphne script, similar to example [scripts/examples/hello-world.daph](../scripts/examples/hello-world.daph), which prints out the running time of a simple operation.
8. [deploy/singularity](singularity)
  - The Singularity image configuration file.

Some associated files in [doc/](../doc/) directory for further reading:

1. [doc/Deploy.md](../doc/Deploy.md) 
  - More documentation about deployment, including tutorial-like explanation examples about how to package, distributively deploy, manage, and execute workloads using DAPHNE.
2. [doc/GettingStarted.md](../doc/GettingStarted.md)
  - Explanation of the switches from comment of PR #335.
3. [doc/development/BuildingDaphne.md](../doc/development/BuildingDaphne.md)
  - The script for full-fledged source code ([src/](../src/)) building - cloning, dependency setup (w/ download), compilation, generation (linking) of the Rarget Runtime(s), and compilation cleanup.

### More Documentation

Refer to [doc/Deploy.md](../doc/Deploy.md) for more documentation about deployment.
