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

This file ([doc/Deploy.md](Deploy.md)) explains deployment of **Daphne system** on HPC (e.g. w/ Slurm), and highlights the excerpts from descriptions of functionalities in [deploy/](../deploy/) directory, mainly [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh):
- compilation of the Singularity image,
- compilation of Daphne (and the Daphne DistributedWorker) within the Singularity image,
- packaging compiled daphne targets,
- packaging compiled daphne targets with user payload as a payload package,
- uploading the payload package to an HPC platform,
- obtaining the connection setup (list of `PEERS` as an environmental variable) from executing Daphne using the Slurm Workload Manager,
- executing daphne main and worker binaries on Slurm `PEERS`,
- collection of logs from daphne execution, and
- cleanup of worker environments and payload deployment.

### Deployment Functionalities

Daphne's distributed system consists of a single coordinator and multiple DistributedWorkers. To execute Daphne in a distributed fashion, first we need to instantiate DistributedWorkers and connect them to the coordinator.
The connection of DistributedWorkers to the coordinator in the Daphne system is achieved through the `PEERS` environmental variable, passed during the deployment. Such deployment is described below. The default ports for worker peers begin at 50000 (`PORTRANGE_BEGIN`) and the list of `PEERS` is generated as `PEERS = ( WORKER1_IP:PORTRANGE_BEGIN, WORKER1_IP:PORTRANGE_BEGIN+1, ..., WORKER2_IP:PORTRANGE_BEGIN, WORKER2_IP:PORTRANGE_BEGIN+1, ... )`.

The DaphneDSL are then run within the Daphne distributed system. Running on the Daphne distributed system does not require any changes to the DaphneDSL code, but it expects to have deployed DistributedWorkers.

The [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh) packages and starts Daphne system on a target HPC platform, and is tailored to the communication required with Slurm and the target HPC platform.

#### Deploying without Slurm support
The packaging and transfer script [deployDistributed.sh](../deploy/deployDistributed.sh) already provides some initial functionality.
Additional specifically upgraded functionalities (with Slurm support and HPC with shared home directory) are in the extended [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh).

The building of the Daphne system to be later started on distributed nodes, can be run through a Singularity container. The Singularity container can be built on the utilized HPC. As the function `deploy` in [deployDistributed.sh](../deploy/deployDistributed.sh) sends and builds executables on each node, which might cause overwrite if the workers use same mounted user storage (e.g. distributed storage attached as home directory), [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh) should be used for such deployments. The latter also automatically generates the environmental variable `PEERS` from Slurm, while for [deployDistributed.sh](../deploy/deployDistributed.sh) the specification of `PEERS` needs to be provided manually.

### How to use DAPHNE Packaging, Distributed Deployment, and Management of Runtime Systems (command deploy-distributed-on-slurm.sh)

This explains how to set up the Distributed Workers on a Deployment Platform, and it also briefly comments on what to do afterwards (how to run, analyse, stop, and clean it).
Commands, with their parameters and arguments, are hence described below for deployment with [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh).


```
Usage: deploy-distributed-on-slurm.sh <options> <command>

Start the DAPHNE distributed deployment on remote machines using Slurm.

These are the options (short and long formats available):
  -h, --help              Print this help message and exit.
  -i SSH_IDENTITY_FILE    Specify OpenSSH identity file (default: private key in ~/.ssh/id_rsa.pub).
  -u, --user SSH_USERNAME Specify OpenSSH username (default: $USER).
  -l, --login SSH_LOGIN_NODE_HOSTNAME     Specify OpenSSH login name hostname (default: localhost).
  -d, --pathToBuild       A path to deploy or where the build is already deployed (default ~/DaphneDistributedWorker can be specified in the script).
  -n, --numcores          Specify number of workers (cores) to use to deploy DAPHNE workers (default: 128).
  -p, --port              Specify DAPHNE deployed port range begin (default: 50000).
  --args ARGS_CS          Specify arguments of a DAPHNE SCRIPT in a comma-separated format.
  -S, --ssh-arg=S         Specify additional arguments S for ssh client (default command: $SSH_COMMAND).
  -C, --scp-arg=C         Specify additional arguments C for scp client (default command: $SCP_COMMAND).
  -R, --srun-arg=R        Specify additional arguments R for srun client.
  -G, --singularity-arg=G Specify additional arguments G for singularity client.

These are the commands that can be executed:
  singularity             Compile the Singularity SIF image for DAPHNE (and transfer it to the target platform).
  build                   Compile DAPHNE codes (daphne, DistributedWorker) using the Singularity image for DAPHNE.
                          It should only be invoked from the code base root directory.
                          It could also be invoked on a target platform after a transfer.
  package                 Create the package image with *.daphne scripts and a compressed build/ directory.
  transfer                Transfers (uploads) a package to the target platform.
  start                   Run workers on remote machines through login node (deploys this script and runs workers).
  workers                 Run workers on current login node through Slurm.
  status                  Get distributed workers' status.
  wait                    Waits until all workers are up.
  stop                    Stops all distributed workers.
  run [SCRIPT [ARGs]]     Run one request on the deployed platform by processing one DAPHNE SCRIPT file (default: /dev/stdin)
                          using optional arguments (ARGs in script format).
  clean                   Cleans (deletes) the package on the target platform.
  deploy                  Deploys everything in one sweep: singularity=>build=>package=>transfer=>start=>wait=>run=>clean.


The default connection to the target platform (HPC) login node is through OpenSSH, configured by default in ~/.ssh (see: man ssh_config).

The default ports for worker peers begin at 50000 (PORTRANGE_BEGIN) and the list of PEERS is generated as:
PEERS = ( WORKER1_IP:PORTRANGE_BEGIN, WORKER1_IP:PORTRANGE_BEGIN+1, ..., WORKER2_IP:PORTRANGE_BEGIN, WORKER2_IP:PORTRANGE_BEGIN+1, ... )

Logs can be found at [pathToBuild]/logs.
```

### Short Examples

The following list presents few examples about how to use the [deploy-distributed-on-slurm.sh](../deploy/deploy-distributed-on-slurm.sh) command.
These comprise more hands-on documentation about deployment, including tutorial-like explanation examples about how to package, distributively deploy, manage, and execute workloads using DAPHNE.

1. Builds the Singularity image and uses it to compile the build directory codes, then packages it.
```shell
./deploy-distributed-on-slurm.sh singularity && ./deploy-distributed-on-slurm.sh build && ./deploy-distributed-on-slurm.sh package
```


2. Transfers a package to the target platform through OpenSSH, using login node HPC, user hpc, and identify key hpc.pub.
```shell
./deploy-distributed-on-slurm.sh --login HPC --user hpc -i ~/.ssh/hpc.pub transfer
```


3. Using login node HPC, accesses the target platform and starts workers on remote machines.
```shell
./deploy-distributed-on-slurm.sh -l HPC start
```


4. Runs one request (script called example-time.daphne) on the deployment using 1024 cores, login node HPC, and default OpenSSH configuration.
```shell
./deploy-distributed-on-slurm.sh -l HPC -n 1024 run example-time.daphne
```


5. Executes one request (DAPHNE script input from standard input) at a running deployed platform, using default singularity/srun configurations.
```shell
./deploy-distributed-on-slurm.sh run
```


6. Deploys once at the target platform through OpenSSH using default login node (localhost), then cleans.
```shell
./deploy-distributed-on-slurm.sh deploy -n 10
```


7. Starts workers at a running deployed platform using custom srun arguments (2 hours dual-core with 10G memory).
```shell
./deploy-distributed-on-slurm.sh workers -R="-t 120 --mem-per-cpu=10G --cpu-bind=cores --cpus-per-task=2"
```


8. Executes a request with custom srun arguments (30 minutes single-core).
```shell
./deploy-distributed-on-slurm.sh run -R="--time=30 --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1"
```


9. Example request job from a pipe.
```shell
cat ../scripts/examples/hello-world.daph | ./deploy-distributed-on-slurm.sh run
```


### Scenario Usage Example

Here is a scenario usage as a longer example demo.

1. Fetch the code from the latest GitHub code repository.
```shell
function compile() {
  git clone --recursive git@github.com:daphne-eu/daphne.git 2>&1 | tee daphne-$(date +%F-%T).log
  cd daphne/deploy
  ./deploy-distributed-on-slurm.sh singularity # creates the Singularity container image
  ./deploy-distributed-on-slurm.sh build   # Builds the daphne codes using the container
}
compile
```


2. Package the built targets (binaries) to packet file `daphne-package.tgz`.
```shell
./deploy-distributed-on-slurm.sh package
```


3. Transfer the packet file `daphne-package.tgz` to `HPC` (Slurm) with OpenSSH key `~/.ssh/hpc.pub` and unpack it.
```shell
./deploy-distributed-on-slurm.sh --login HPC --user $USER -i ~/.ssh/hpc.pub transfer 
```
E.g., for EuroHPC Vega, use the instance, if your username matches the one at Vega and the key is `~/.ssh/hpc.pub`:
```shell
./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub transfer
```


4. Start the workers from the local computer by logging into the HPC login node:
```shell
./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub start
```


5. Starting a main target on the HPC (Slurm) and connecting it with the started workers, to execute payload from the stream.
```shell
cat ../scripts/examples/hello-world.daph | ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub run 
```


6. Starting a main target on the HPC (Slurm) and connecting it with the started workers, to execute payload from a file.
```shell
./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub run example-time.daphne
```


7. Stopping all workers on the HPC (Slurm).
```shell
./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub stop
```


8. Cleaning the uploaded targets from the HPC login node.
```shell
./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub clean
```
