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


## Overview

This directory [deploy/](/deploy/) can be used to deploy the Daphne System.
With these scripts one can:
- build  the Daphne System (using [build.sh](/build.sh)),
- package,
- deliver and install to a deployment platform (e.g. HPC) and
- utilize the resources of multiple machines/nodes.
- It can also be used to just try out DAPHNE on a single machine.

Once deployed, Daphne system consists of multiple `DistributedWorker`s and a single `coordinator` who is responsible for handling a distributed execution.

### Where to Start

- [deployDistributed.sh](deployDistributed.sh) can be used to manually deploy using only SSH. When executed without parameters, it prints out the help message. 
- [deploy-distributed-on-slurm.sh](deploy-distributed-on-slurm.sh) can be used for environments with Slurm tool. When executed without parameters, it prints out the help message.
## Deployment Scheme

DAPHNE Deployment Scheme encompasses the following:

- A Compilation node (where the Daphne System will be compiled)
  <!-- - OpenSSH connection to the Deployment Platform
  - (optional) Internet connection to fetch the source code and dependencies -->
- Deployment Platform (e.g. an HPC with SLURM support)
  - Login Node (or, other type of access)
    - HPC Task Submission interface (e.g. SLURM)
  - Compute Node(s)
    <!-- - Interface for provisioned tasks from SLURM -->
    - DAPHNE `coordinator`
    - DAPHNE `DistributedWorker`s

```
                    DAPHNE Deployment Scheme

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
| | Deployment Platform (e.g. an HPC with SLURM support)                             | |
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
| |  | - Running Tasks          |     |   (e.g. 128+)            |     |           | | |
| |  |   - `coordinator`        |     | {DistributedWorker (DW)} |     | DWs       | | |
| |  |   - (optional: more DWs) |     |   (e.g. DWs 1..128)      |     |           | | |
| |  +--------------------------+     +--------------------------+     +-----------+ | |
| |                                                                                  | |
| +----------------------------------------------------------------------------------+ |
|                                                                                      |
+--------------------------------------------------------------------------------------+
```

## Deployment scripts

This directory includes a set of **bash scripts** providing support for:
- packaging/virtualization of the deployment (installation) package,
- containerized packaging,
- virtualized installation,
- managed deployment,
- deployment of the ˙daphne˙ executable,
- starting and managing Daphne processes within containerized environments (schedule  and  execute remotely SLURM tasks), and
- stopping and cleaning of a deployment.

## List of Files in this Directory

1. This short [README](README.md) file to explain directory structure and point to more documentation at [Deploy](/doc/Deploy.md).
2. A [script](build-daphne-singularity-image.sh) that builds the "daphne.sif" singularity image from the [Docker image](/containers/README.md)
daphneeu/daphne-dev
3. [deploy-distributed-on-slurm](deploy-distributed-on-slurm.sh) script allows the user to deploy DAPHNE with SLURM.
4. [deployDistributed](deployDistributed.sh) script builds and sends DAPHNE to remote machines manually with SSH (no tools like Slurm needed).
5. [example-time.daphne](example-time.daphne) Daphne example script which prints out the running time of a simple operation.
6. The [Singularity image](singularity) configuration file.

### More Documentation

1. [Documentation about deployment](/doc/Deploy.md), including tutorial-like explanation examples about how to package, distributively deploy, manage, and execute workloads using DAPHNE.
2. [Getting started guide](/doc/GettingStarted.md)
3. [Bulding the Daphne System](/doc/development/BuildingDaphne.md)
