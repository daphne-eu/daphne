#!/bin/bash

# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an extract from distributed execution deployment scripts, from the
# experimental branch deploy-Vega, to support deployment with SLURM.
# This script allows the user to deploy DAPHNE through SLURM, including:
# 1 - compilation of the Singularity image,
# 2 - compilation of the daphne main and worker codes within the Singularity image
# 3 - packaging compiled daphne codes
# 4 - packaging compiled daphne codes with user payload as a payload package
# 5 - uploading the payload package to an HPC platform
# 6 - obtaining the list of PEERS from SLURM
# 7 - executing daphne main and worker binaries on SLURM PEERS
# 8 - collection of logs from daphne execution
# 9 - cleanup of workers and payload deployment
# The difference of this script from deploy-distributed-on-slurm.sh is that
# while packaging and executing on a target HPC platform, it is tailored to
# the communication required with SLURM and the target HPC platform.
#
# Specific description of functionality differences with deploy-distributed-on-slurm.sh:
#
# - the build of the daphne main and worker node executables is executed
#   through a Singularity container that is built on the utilized HPC,
#   while the function "deploy" in deployDistriuted.sh sends and builds
#   executables on each node, which might cause overwrite if the workers use same
#   mounted user storage (e.g. distributed storage attached as home directory)
#
# - the list of PEERS is not defined by the user but obtained from SLURM
#   (in deployDistriuted.sh, the user supplies PEERS as an argument)
#
# - the support for single request deployment, run, and cleanup is provided
#



#*****************************************************************************
# SSH Configurations to access the HPC platform
#*****************************************************************************

SSH_LOGIN_NODE_HOSTNAME=
SSH_SSH_IDENTITY_FILE=
SSH_USERNAME=$USER
SSH_PORT=22


#*****************************************************************************
# DAPHNE deployment default parameters
#*****************************************************************************

NUMCORES=128
DAPHNE_SCRIPT=execute.daphne

#*****************************************************************************
# Parameterization of the HPC platform specifics (optional)
#*****************************************************************************

PORTRANGE_BEGIN=50000
PATH_TO_DEPLOY_BUILD=~/DaphneDistributedWorker
PEERS= # the "PEERS" variable is obtained using SLURM below


#*****************************************************************************
#           DO (not) EDIT BELOW THIS LINE
#*****************************************************************************

SSH_COMMAND="ssh ${SSH_IDENTITY_FILE:+-i $SSH_IDENTITY_FILE} ${SSH_PORT:+-p $SSH_PORT}"
SCP_COMMAND="scp ${SSH_IDENTITY_FILE:+-i $SSH_IDENTITY_FILE} ${SSH_PORT:+-P $SSH_PORT}"

# Stop immediately if any command fails.
set -e


# ****************************************************************************
# Builds the image for a Singularity container, then transfers it to the 
# target platform.
# ****************************************************************************

function buildAndTransferSingularityContainerImage {
    ./deploy/build-daphne-singularity-image.sh

    $SCP_COMMAND daphne.sif $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME:$PATH_TO_DEPLOY_BUILD
}


# ****************************************************************************
# Builds daphne code and puts it in a time-stamped build directory. 
# The build is called using the daphne singularity container "daphne.sif" 
# created from the DAPHNE Docker image.
# The daphne and DistributedWorker are both built.
# ****************************************************************************

function buildWithSingularityContainer {
    time singularity exec daphne.sif ./build.sh
    time singularity exec daphne.sif ./build.sh --target DistributedWorker
    TIME_BUILT=$(date  +%F-%T)
    mv build build_${TIME_BUILT}
}


# ****************************************************************************
# Packaging of files (payload) to be sent to a remote machine:
# - daphne built (build/) and 
# - user code (all *.daphne scripts in the current working directory).
# The step of packaging using same compilation (build_*) is hence reusable.
# ****************************************************************************

function packageBuiltDaphnePayload {
    echo "Packaging latest files for DaphneDistributedWorker deployment..."
    (
    tar cvzf build.tgz build/
    tar cvzf daphne-worker.tgz build.tgz *.daphne deploy-distributed-on-slurm.sh
    ) | awk '{printf("\r%-100s      ", substr($0, -1, 100));}'
}


# ****************************************************************************
# Deploys: copies the package to the target platform using SCP and extracts it.
# The target deployment platform (HPC) is configured in SSH/SCP Configuration.
# All distributed workers are expected to be able to access an extracted package.
# For transfer, scp is used, since rsync might already be used for "daphne.sif".
# ****************************************************************************

function deployPackageToTargetPlatform {
    $SCP_COMMAND daphne-worker.tgz $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME:$PATH_TO_DEPLOY_BUILD

    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME <<EOF
cd $PATH_TO_DEPLOY_BUILD
tar xvf daphne-worker.tgz
exit
EOF
}


#*****************************************************************************
# Remotely start workers on remote machines (calls -R, recursively on login node for SLURM and containers).
# This script itself is expected to be present at the target platform,
# packaged with packageBuiltDaphnePayload and transferred with deployPackageToTargetPlatform.
#*****************************************************************************

function RemotelyStartWorkers {
    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME <<EOF
cd $PATH_TO_DEPLOY_BUILD
./deploy-distributed-on-slurm.sh -R
exit
EOF
}


#*****************************************************************************
# Start workers on remote machines through SLURM using Singularity containers.
# This can be called from RemotelyStartWorkers.
#*****************************************************************************

function StartWorkersInContainersOnSLURM {
    mkdir -p logs
    rm logs/* 2>/dev/null # clean workerlist and other content

    srun -J DAPHNEworkers --time=119 --mem-per-cpu=10G ${DEMO_USE_CUDA} --cpu-bind=cores --cpus-per-task=2 -n $NUMCORES \
        bash -c 'singularity exec daphne.sif \
                    build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID )) \
                         > logs/OUTPUT.$(hostname):$(( 50000 + SLURM_LOCALID )) \
                        2>&1 \
                        & echo \$! > logs/PID.$(hostname):$(( 50000 + SLURM_LOCALID ))' &
}


#*****************************************************************************
# Get worker's status
#*****************************************************************************

function WorkersStatus {
    PEERS=$(cd logs; echo OUTPUT.* | sed -e 's/OUTPUT.//g' -e 's/ /,/g')
    echo "PEERS list: $PEERS"
    
    [ $(cd logs; ls -1 OUTPUT.* 2>/dev/null | wc -l) -ge $NUMCORES ] && echo All up.
    
    echo -e "\nInfo about the daphnec build/ dir is:"
    cat $PATH_TO_DEPLOY_BUILD/build/git_source_status_info
}

#*****************************************************************************
# Waits for all workers to be run through SLURM.
#*****************************************************************************

function waitAllWorkersUp {
    until [ $(cd logs; ls -1 OUTPUT.* 2>/dev/null | wc -l) -ge $NUMCORES ]
    do
            echo -n .
            sleep 1
    done
    
    echo -e "\nSuccessfully spawned N new distributed worker daemons (see queue below), N=" $NUMCORES
    squeue -u $(whoami) # print the generated worker list
}
    
#*****************************************************************************
# Run one request (.daphne script) on an already deployed distributed platform.
#*****************************************************************************

function RunOneRequest {
      time srun --time=30 ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
        singularity exec daphne.sif bash -c \
            'DISTRIBUTED_WORKERS='${WORKERS}" $PATH_TO_DEPLOY_BUILD"'/build/bin/daphne '$DAPHNE_SCRIPT
}


#*****************************************************************************
# Kill workers
#*****************************************************************************

function KillWorkersOnSLURM {
    scancel -n DAPHNEworkers
}


#*****************************************************************************
# Run one request (.daphne script) on an already deployed distributed platform.
#*****************************************************************************

function DeploymentClean {
    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME <<EOF
rm -rf $PATH_TO_DEPLOY_BUILD
exit
EOF
}


# ****************************************************************************
# A more general deploy function, envoking all necessary steps to deploy a 
# working DAPHNE deployment workers set: builds with Singularity, packages,
# copies the package, executes, and cleans a deployment at the HPC platform.
# ****************************************************************************

function DeployEverythingOnce {
    buildAndTransferSingularityContainerImage
    
    buildWithSingularityContainer
    
    packageBuiltDaphnePayload
    
    deployPackageToTargetPlatform
    
    RemotelyStartWorkers
    
    waitAllWorkersUp
    
    WorkersStatus

    RunOneRequest
    
    KillWorkersOnSLURM
    
    DeploymentClean
}


#*****************************************************************************
#*****************************************************************************
# General functions below as in deployDistributed.sh, tailored for this script:
# printHelp, parsePeers, parseArguments.
#*****************************************************************************
#*****************************************************************************


#*****************************************************************************
# Help message
#*****************************************************************************

function printHelp {
    echo "Start the DAPHNE distributed deployment on remote machines using SLURM."
    echo "Usage: $0 [-h|--help] [-i] [--user] [-l|--login] [--deploy] [--pathToBuild] [-r| --run] [-n ] [-p] [-s| --status] [--kill] [ -D | --daphneScript] [ -S ]"
    echo ""
    echo "The default connection to the target platform (HPC) login node is through OpenSSH, configured by default in ~/.ssh (see: man ssh_config)."
    echo ""
    echo "The default ports for worker peers begin at 50000 (PORTRANGE_BEGIN) and the list of PEERS is generated as:"
    echo "PEERS = ( WORKER1_IP:PORTRANGE_BEGIN, WORKER1_IP:PORTRANGE_BEGIN+1, ..., WORKER2_IP:PORTRANGE_BEGIN, WORKER2_IP:PORTRANGE_BEGIN+1, ... )"
    echo ""
    echo "Logs can be found at [pathToBuild]/logs."
    echo ""
    echo "--deploy:"
    echo "This includes downloading and building all required third-party "
    echo "material (if necessary), for building the DistributedWorker."
    echo "You should only invoke it from the prototype's root directory."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help              Print this help message and exit."
    echo "  -i SSH_IDENTITY_FILE    Specify OpenSSH identity file (default: default ssh private key)."
    echo "  --user SSH_USERNAME     Specify OpenSSH username (default: \$USER)."
    echo "  --login SSH_LOGIN_NODE_HOSTNAME     Specify OpenSSH login name hostname."
    echo "  --deploy                Compress and deploy build folder to remote machines."
    echo "  --pathToBuild           A path to deploy or where the build is already deployed (default ~/DaphneDistributedWorker can be specified in the script)."
    echo "  -r, --run               Run workers on remote machines through login node."
    echo "  -R, --runOnSLURM        Run workers on current login node through SLURM."
    echo "  -n, --numcores          Specify number of workers (cores) to use to deploy DAPHNE workers."
    echo "  -p, --port              Specify DAPHNE deployed port range begin (default 50000)."
    echo "  -s, --status            Get distributed workers' status."
    echo "  --kill                  Kill all distributed workers."
    echo "  -D DAPHNE_SCRIPT        Filename of the daphne script to run (e.g. execute.daphne)."
    echo "  -S                      Run one request on the deployed platform (process one DAPHNE_SCRIPT script)."
    echo ""
    echo "Example:"
    echo "$0 -l HPC --user hpc -i ~/.ssh/hpc.pub --deploy         Deploys over OpenSSH at login node HPC using user hpc and key hpc.pub to the target."
    echo "$0 -l HPC -r -n 1024 -D example-time.daphne -S          Runs one request (script called example-time.daphne) on the deployment using 1024 cores, using login node HPC and default OpenSSH configuration."
    echo "$0 -l HPC -S                                            Executes one request to a running deployed platform, using login node HPC and default OpenSSH configuration and the default script filename execute.daphne."
}


#*****************************************************************************
# Parse arguments
#*****************************************************************************

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -h|--help)
        printHelp
        shift 1
        exit 0
        ;;
    -i)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_IDENTITY_FILE=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi        
        ;;
    --user)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_USERNAME=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi        
        ;;
    --login)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_LOGIN_NODE_HOSTNAME=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi        
        ;;
    -r| --run)
        START_WORKERS_FLAG=TRUE
        shift 1
        ;;
    -R| --runOnSLURM)
        START_WORKERS_SLURM_FLAG=TRUE
        shift 1
        ;;        
    -n|--numcores)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            NUMCORES=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi
        ;;
    -p|--port)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PORTRANGE_BEGIN=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi
        ;;
    --deploy)
        DEPLOY_FLAG=TRUE
        shift 1
        ;;
    --pathToBuild)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PATH_TO_BUILD=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi
        ;;
    -s| --status)
        WORKERS_STATUS_FLAG=TRUE
        shift 1
        ;;
    -D|--daphneScript)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            DAPHNE_SCRIPT=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi
        ;;
    --kill)
        KILL_WORKERS_FLAG=TRUE
        shift 1
        ;;
    -*|--*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
    *) # preserve positional arguments
        PARAMS="$PARAMS $1"
        shift
        ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

if [[ ! -n $PATH_TO_DEPLOY_BUILD ]]; then
    echo "You must specifcy where the build is located."
    exit 1
fi

if [[ ! -n $SSH_USERNAME ]]; then
    echo "Please configure the client OpenSSH username (inside the script)."
    exit 1
fi

if [[ -n $DEPLOY_FLAG ]]; then   
    DeployEverythingOnce
    exit 0
fi

if [[ -n $START_WORKERS_FLAG ]]; then
    RemotelyStartWorkers
    exit 0
fi

if [[ -n $START_WORKERS_SLURM_FLAG ]]; then
    StartWorkersInContainersOnSLURM
    exit 0
fi

if [[ -n $WORKERS_STATUS_FLAG ]]; then
    WorkersStatus
    exit 0
fi

if [[ -n $KILL_WORKERS_FLAG ]]; then
    KillWorkersOnSLURM
    exit 0
fi

printHelp
