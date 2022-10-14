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

# DAPHNE Packaging, Distributed Deployment, and Management

### Overview

This file ([doc/Deploy.md](Deploy.md)) explains deployment of **DAPHNE runtime systems** on HPC (e.g. w/ SLURM), and highlights the excerpts from descriptions of functionalities in [deploy/](../deploy/) directory, mainly [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh):
- compilation of the Singularity image,
- compilation within the Singularity image, of the codes for `daphne` main target and worker target,
- packaging compiled daphne targets,
- packaging compiled daphne targets with user payload as a payload package,
- uploading the payload package to an HPC platform,
- obtaining the list of `PEERS`from SLURM,
- executing daphne main and worker binaries on SLURM `PEERS`,
- collection of logs from daphne execution, and
- cleanup of worker environments and payload deployment.

### Deployment Functionalities for SLURM

The [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh) packages and starts targets on a target HPC platform, and is tailored to the communication required with SLURM and the target HPC platform. While the packaging and transfer script [deployDistributed.sh](../deploy/deployDistributed.sh) already provides some functionality, the specific upgraded functionalities in the extended [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh) compared to [deployDistributed.sh](../deploy/deployDistributed.sh) are:
- The building of the `daphne` main and worker targets to be later started on distributed nodes, can be run through a Singularity container. The Singularity container can be built on the utilized HPC. Otherwise, the function `deploy` in [deployDistributed.sh](../deploy/deployDistributed.sh) sends and builds executables on each node, which might cause overwrite if the workers use same mounted user storage (e.g. distributed storage attached as home directory).
- The list of `PEERS` is not defined by the user but obtained from SLURM (in `deployDistributed.sh`, the user supplies `PEERS` as an argument).
- Specifying `SLURM` running time for single DAPHNE main target duration is provided (with `RunOneRequest`).
- Cleanup support is added.
    
### How to use DAPHNE Packaging, Distributed Deployment, and Management of Runtime Systems

TBD
