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

# Deploying

DAPHNE Packaging, Distributed Deployment, and Management

## Overview

This file explains the deployment of the **Daphne system**, on HPC with SLURM or manually through SSH, and highlights the excerpts from descriptions of functionalities in [deploy/](/deploy/) directory (mostly [deploy-distributed-on-slurm.sh](/deploy/deploy-distributed-on-slurm.sh)):

- compilation of the Singularity image,
- compilation of Daphne (and the Daphne DistributedWorker) within the Singularity image,
- packaging compiled Daphne,
- packaging compiled Daphne with user payload as a payload package,
- uploading the payload package to an HPC platform,
- starting and managing DAPHNE workers on HPC platforms using SLURM,
- executing DAPHNE on HPC using SLURM,
- collection of logs from daphne execution, and
- cleanup of worker environments and payload deployment.

## Background

Daphne's distributed system consists of a single coordinator and multiple DistributedWorkers (you can read more about Distributed DAPHNE [here](DistributedRuntime.md#Background)).  For now, in order to execute Daphne in a distributed fashion, we need to deploy DistributedWorkers manually. Coordinator gets the worker's addresses
through an environmental variable.

`deployDistributed.sh` manually connects to machines with SSH and starts up DistributedWorker processes. On the other hand the [deploy-distributed-on-slurm.sh](/deploy/deploy-distributed-on-slurm.sh) packages and starts Daphne on a target HPC platform, and is tailored to the communication required with Slurm and the target HPC platform.

## Deploying without Slurm support

**`deployDistributed.sh`** can be used to manually connect to a list of machines and remotely start up workers, get status of running workers or terminate distributed worker processes. This script depends only on an SSH client/server and does not require any use of a resource management tool (e.g. SLURM). With this script you can:

- build and deploy DistributedWorkers to remote machines
- start workers
- check status of running workers
- kill workers

Workers' own IPs and ports to listen to, can be specified inside the script, or with `--peers [IP[:PORT]],[IP[:PORT]],...`. Default port for all workers is 50000, but this can also be specified inside the script or with `-p,--port PORT`. If running on same machine (e.g. localhost), different ports must be specified.

With `--deploy` the script builds `DistributedWorker` executable (`./build.sh --target DistributedWorker`), compresses `build`, `lib` and `bin` folders and uses `scp` and `ssh` to send and decompress at remote machines, inside the directory specified by `--pathToBuild` (default `~/DaphneDistributedWorker/`). If running workers on localhost, `PATH_TO_BUILD` can be set `/path/to/daphne` and provided `DistributedWorker` is built, `--deploy` is not nessecary.

Ssh username must be specified inside the script. For now the script assumes all remote machines can be accessed with the same `username`, `id_rsa` key and ssh port (default 22).

Usage example:

```bash
# deploy distributed
$ ./deployDistributed.sh --help
$ ./deployDistributed.sh --deploy --pathToBuild /path/to/dir --peers localhost:5000,localhost:5001
$ ./deployDistributed.sh -r # (Uses default peers and path/to/build/ to start workers)
```  

## Deploying with Slurm support

Building the Daphne system (to be later deployed on distributed nodes) can be done with a Singularity container. The Singularity container can be built on the utilized HPC. [deployDistributed.sh](/deploy/deployDistributed.sh) sends executables on each node, assuming there are different storages for each node. This might cause unnecessary overwrites if the workers use same mounted user storage (e.g. HPC environments with distributed storages). Instead [deploy-distributed-on-slurm.sh](/deploy/deploy-distributed-on-slurm.sh) should be used for such cases. The latter also automatically generates the environmental variable `PEERS` from Slurm.

### How to use `deploy-distributed-on-slurm.sh` for DAPHNE Packaging, Distributed Deployment, and Management using Slurm

This explains how to set up the Distributed Workers on a HPC platform, and it also briefly comments on what to do afterwards (how to run, manage, stop, and clean it).
Commands, with their parameters and arguments, are hence described below for deployment with [deploy-distributed-on-slurm.sh](/deploy/deploy-distributed-on-slurm.sh).

```shell
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
  --args ARGS_CS          Specify arguments of a DaphneDSL SCRIPT in a comma-separated format.
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
  run [SCRIPT [ARGs]]     Run one request on the deployed platform by processing one DaphneDSL SCRIPT file (default: /dev/stdin)
                          using optional arguments (ARGs in script format).
  clean                   Cleans (deletes) the package on the target platform.
  deploy                  Deploys everything in one sweep: singularity=>build=>package=>transfer=>start=>wait=>run=>clean.


The default connection to the target platform (HPC) login node is through OpenSSH, configured by default in ~/.ssh (see: man ssh_config).

The default ports for worker peers begin at 50000 (PORTRANGE_BEGIN) and the list of PEERS is generated as:
PEERS = ( WORKER1_IP:PORTRANGE_BEGIN, WORKER1_IP:PORTRANGE_BEGIN+1, ..., WORKER2_IP:PORTRANGE_BEGIN, WORKER2_IP:PORTRANGE_BEGIN+1, ... )

Logs can be found at [pathToBuild]/logs.
```

### Short Examples

The following list presents few examples about how to use the [deploy-distributed-on-slurm.sh](/deploy/deploy-distributed-on-slurm.sh) command.

These comprise more hands-on documentation about deployment, including tutorial-like explanation examples about how to package, distributively deploy, manage, and execute workloads using DAPHNE.

1. Builds the Singularity image and uses it to compile the build directory codes, then packages it.

    ```shell
    ./deploy-distributed-on-slurm.sh singularity && ./deploy-distributed-on-slurm.sh build && ./deploy-distributed-on-slurm.sh package
    ```

1. Transfers a package to the target platform through OpenSSH, using login node HPC, user hpc, and identify key hpc.pub.

    ```shell
    ./deploy-distributed-on-slurm.sh --login HPC --user hpc -i ~/.ssh/hpc.pub transfer
    ```

1. Using login node HPC, accesses the target platform and starts workers on remote machines.

    ```shell
    ./deploy-distributed-on-slurm.sh -l HPC start
    ```

1. Runs one request (script called example-time.daphne) on the deployment using 1024 cores, login node HPC, and default OpenSSH configuration.

    ```shell
    ./deploy-distributed-on-slurm.sh -l HPC -n 1024 run example-time.daphne
    ```

1. Executes one request (DaphneDSL script input from standard input) at a running deployed platform, using default singularity/srun configurations.

    ```shell
    ./deploy-distributed-on-slurm.sh run
    ```

1. Deploys once at the target platform through OpenSSH using default login node (localhost), then cleans.

    ```shell
    ./deploy-distributed-on-slurm.sh deploy -n 10
    ```

1. Starts workers at a running deployed platform using custom srun arguments (2 hours dual-core with 10G memory).

    ```shell
    ./deploy-distributed-on-slurm.sh workers -R="-t 120 --mem-per-cpu=10G --cpu-bind=cores --cpus-per-task=2"
    ```

1. Executes a request with custom srun arguments (30 minutes single-core).

    ```shell
    ./deploy-distributed-on-slurm.sh run -R="--time=30 --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1"
    ```

1. Example request job from a pipe.

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

1. Package the built targets (binaries) to packet file `daphne-package.tgz`.

    ```shell
    ./deploy-distributed-on-slurm.sh package
    ```

1. Transfer the packet file `daphne-package.tgz` to `HPC` (Slurm) with OpenSSH key `~/.ssh/hpc.pub` and unpack it.

    ```shell
    ./deploy-distributed-on-slurm.sh --login HPC --user $USER -i ~/.ssh/hpc.pub transfer 
    ```

    E.g., for EuroHPC Vega, use the instance, if your username matches the one at Vega and the key is `~/.ssh/hpc.pub`:

    ```shell
    ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub transfer
    ```

1. Start the workers from the local computer by logging into the HPC login node:

    ```shell
    ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub start
    ```

1. Starting a main target on the HPC (Slurm) and connecting it with the started workers, to execute payload from the stream.

    ```shell
    cat ../scripts/examples/hello-world.daph | ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub run 
    ```

1. Starting a main target on the HPC (Slurm) and connecting it with the started workers, to execute payload from a file.

    ```shell
    ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub run example-time.daphne
    ```

1. Stopping all workers on the HPC (Slurm).

    ```shell
    ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub stop
    ```

1. Cleaning the uploaded targets from the HPC login node.

    ```shell
    ./deploy-distributed-on-slurm.sh --login login.vega.izum.si --user $USER -i ~/.ssh/hpc.pub clean
    ```
