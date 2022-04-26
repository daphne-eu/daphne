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
# - an expanded format of parameters, commands, and arguments
#



#*****************************************************************************
# SSH Configurations to access the HPC platform
#*****************************************************************************

SSH_LOGIN_NODE_HOSTNAME=localhost
SSH_IDENTITY_FILE=~/.ssh/id_rsa.pub
SSH_USERNAME=$USER
SSH_PORT=22


#*****************************************************************************
# DAPHNE deployment default parameters
#*****************************************************************************

NUMCORES=128
DAPHNE_SCRIPT_AND_PARAMS=/dev/stdin


#*****************************************************************************
# Parameterization of the HPC platform specifics (optional)
#*****************************************************************************

PORTRANGE_BEGIN=50000
PATH_TO_DEPLOY_BUILD=./DaphneDistributedWorker
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

function BuildAndTransferSingularityContainerImage {
    ./deploy/build-daphne-singularity-image.sh

    $SCP_COMMAND daphne.sif $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME:$PATH_TO_DEPLOY_BUILD
}


# ****************************************************************************
# Builds daphne code and puts it in a time-stamped build directory. 
# The build is called using the daphne singularity container "daphne.sif" 
# created from the DAPHNE Docker image.
# The daphne and DistributedWorker are both built.
# ****************************************************************************

function BuildWithSingularityContainer {
    time singularity $SINGULARITY_ARG exec daphne.sif ./build.sh
    time singularity $SINGULARITY_ARG exec daphne.sif ./build.sh --target DistributedWorker
    TIME_BUILT=$(date  +%F-%T)
    mv build build_${TIME_BUILT}
}


# ****************************************************************************
# Packaging of files (payload) to be sent to a remote machine:
# - daphne built (build/) and 
# - user code (all *.daphne scripts in the current working directory).
# The step of packaging using same compilation (build_*) is hence reusable.
# ****************************************************************************

function PackageBuiltDaphnePayload {
    echo "Packaging latest files for DaphneDistributedWorker deployment..."
    (
    tar cvzf build.tgz build/
    cp $0 deploy-distributed-on-slurm.sh
    chmod 755 deploy-distributed-on-slurm.sh
    tar cvzf daphne-package.tgz build.tgz *.daphne deploy-distributed-on-slurm.sh
    ) | awk '{printf("\r%-100s      ", substr($0, -1, 100));}END{print "";}'
}


# ****************************************************************************
# Deploys: copies the package to the target platform using SCP and extracts it.
# The target deployment platform (HPC) is configured in SSH/SCP Configuration.
# All distributed workers are expected to be able to access an extracted package.
# For transfer, scp is used, since rsync might already be used for "daphne.sif".
# ****************************************************************************

function TransferPackageToTargetPlatform {
    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME mkdir -p $PATH_TO_DEPLOY_BUILD

    $SCP_COMMAND daphne-package.tgz $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME:$PATH_TO_DEPLOY_BUILD

    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME <<EOF
cd $PATH_TO_DEPLOY_BUILD
tar xvf daphne-package.tgz
exit
EOF
}


#*****************************************************************************
# Remotely start workers on remote machines (argument "workers", on login node for SLURM and containers).
# This script itself is expected to be present at the target platform,
# packaged with PackageBuiltDaphnePayload and transferred with TransferPackageToTargetPlatform.
#*****************************************************************************

function RemotelyStartWorkers {
    # forward the parameters to remote invocation
    PARAMS_REMOTE="$PARAMS"
    if [[ -n "$SRUN_ARG" ]]; then   
        PARAMS_REMOTE="-R=\"$SRUN_ARG\" $PARAMS_REMOTE"
    fi
    if [[ -n "$SINGULARITY_ARG" ]]; then   
        PARAMS_REMOTE="-G=\"$SINGULARITY_ARG\" $PARAMS_REMOTE"
    fi
    if [[ -n "$NUMCORES" ]]; then   
        PARAMS_REMOTE="-n $NUMCORES $PARAMS_REMOTE"
    fi
    if [[ -n "$PORTRANGE_BEGIN" ]]; then   
        PARAMS_REMOTE="-p $PORTRANGE_BEGIN $PARAMS_REMOTE"
    fi
    
    $SSH_COMMAND $SSH_USERNAME@$SSH_LOGIN_NODE_HOSTNAME <<EOF
cd $PATH_TO_DEPLOY_BUILD
./deploy-distributed-on-slurm.sh workers -d . $PARAMS_REMOTE
exit
EOF
}


#*****************************************************************************
# Start workers on remote machines through SLURM using Singularity containers.
# This can be called from RemotelyStartWorkers.
#*****************************************************************************

function StartWorkersInContainersOnSLURM {
    mkdir -p logs
    rm -f logs/* 2>/dev/null # clean workerlist and other content

    srun -J DAPHNEworkers $SRUN_ARG -n $NUMCORES \
        bash -c 'singularity '$SINGULARITY_ARG' exec daphne.sif \
                    build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID )) \
                         > logs/OUTPUT.$(hostname):$(( 50000 + SLURM_LOCALID )) \
                        2>&1 \
                        & echo $! > logs/PID.$(hostname):$(( 50000 + SLURM_LOCALID ))' &
}


#*****************************************************************************
# Get worker's status
#*****************************************************************************

function WorkersStatus {
    WORKERS=$(cd logs; echo OUTPUT.* | sed -e 's/OUTPUT.//g' -e 's/ /,/g')
    echo "WORKERS list: $WORKERS"
    
    [ $(cd logs; ls -1 OUTPUT.* 2>/dev/null | wc -l) -ge $NUMCORES ] && echo All up.
    
    echo -e "\nInfo about the DAPHNE build/ dir is:"
    cat build/git_source_status_info
}


#*****************************************************************************
# Waits for all workers to be run through SLURM.
#*****************************************************************************

function WaitAllWorkersUp {
    until [ $(cd logs; ls -1 OUTPUT.* 2>/dev/null | wc -l) -ge $NUMCORES ]
    do
            echo -n .
            sleep 1
    done
    
    echo -e "\nSuccessfully spawned N new distributed worker daemons (see queue below), N=" $NUMCORES
    squeue -u $(whoami) # print the generated worker list
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

function RunOneRequest {
    WORKERS=$(cd logs; echo OUTPUT.* | sed -e 's/OUTPUT.//g' -e 's/ /,/g')
    time srun $SRUN_ARG \
        singularity $SINGULARITY_ARG exec daphne.sif bash -c \
            "DISTRIBUTED_WORKERS=${WORKERS} build/bin/daphne $ARGS_CS $DAPHNE_SCRIPT_AND_PARAMS"
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
# A more general deploy function, evoking all necessary steps to deploy a 
# working DAPHNE deployment workers set: builds with Singularity, packages,
# copies the package, executes, and cleans a deployment at the HPC platform.
# ****************************************************************************

function DeployEverythingHere {
    # building locally or at target, but supporting transfer over OpenSSH
    BuildAndTransferSingularityContainerImage
    
    BuildWithSingularityContainer
    
    PackageBuiltDaphnePayload
    
    TransferPackageToTargetPlatform

    # starting remotely at the platform 
    
    RemotelyStartWorkers
    
    
    # for evocation of the command at target site below here:
  
    WaitAllWorkersUp
    
    WorkersStatus

    RunOneRequest
    
    KillWorkersOnSLURM
    
    # cleanup using OpenSSH
    
    DeploymentClean
}



#*****************************************************************************
#*****************************************************************************
# General functions below as in deployDistributed.sh, tailored for this script,
# with added format for commands and additional parameters and arguments.
#*****************************************************************************
#*****************************************************************************


#*****************************************************************************
# Help message
#*****************************************************************************

function PrintHelp {
    echo "Usage: $0 <options> <command>"
    echo ""
    echo "Start the DAPHNE distributed deployment on remote machines using SLURM."
    echo ""
    echo "These are the options (short and long formats available):"
    echo "  -h, --help              Print this help message and exit."
    echo "  -i SSH_IDENTITY_FILE    Specify OpenSSH identity file (default: private key in ~/.ssh/id_rsa.pub)."
    echo "  -u, --user SSH_USERNAME Specify OpenSSH username (default: \$USER)."
    echo "  -l, --login SSH_LOGIN_NODE_HOSTNAME     Specify OpenSSH login name hostname (default: localhost)."
    echo "  -d, --pathToBuild       A path to deploy or where the build is already deployed (default ~/DaphneDistributedWorker can be specified in the script)."
    echo "  -n, --numcores          Specify number of workers (cores) to use to deploy DAPHNE workers (default: 128)."
    echo "  -p, --port              Specify DAPHNE deployed port range begin (default: 50000)."
    echo "  --args ARGS_CS          Specify arguments of a DAPHNE SCRIPT in a comma-separated format."
    echo "  -S, --ssh-arg=S         Specify additional arguments S for ssh client (default command: $SSH_COMMAND)."
    echo "  -C, --scp-arg=C         Specify additional arguments C for scp client (default command: $SCP_COMMAND)."
    echo "  -R, --srun-arg=R        Specify additional arguments R for srun client."
    echo "  -G, --singularity-arg=G Specify additional arguments G for singularity client."
    echo ""
    echo "These are the commands that can be executed:"
    echo "  singularity             Compile the Singularity SIF image for DAPHNE (and transfer it to the target platform)."
    echo "  build                   Compile DAPHNE codes (daphne, DistributedWorker) using the Singularity image for DAPHNE."
    echo "                          It should only be invoked from the code base root directory."
    echo "                          It could also be invoked on a target platform after a transfer."
    echo "  package                 Create the package image with *.daphne scripts and a compressed build/ directory."
    echo "  transfer                Transfers (uploads) a package to the target platform."
    echo "  start                   Run workers on remote machines through login node (deploys this script and runs workers)."
    echo "  workers                 Run workers on current login node through SLURM."
    echo "  status                  Get distributed workers' status."
    echo "  wait                    Waits untill all workers are up."
    echo "  stop                    Stops all distributed workers."
    echo "  run [SCRIPT [ARGs]]     Run one request on the deployed platform by processing one DAPHNE SCRIPT file (default: /dev/stdin)"
    echo "                          using optional arguments (ARGs in script format)."
    echo "  clean                   Cleans (deletes) the package on the target platform."
    echo "  deploy                  Deploys everything in one sweep: singularity=>build=>package=>transfer=>start=>wait=>run=>clean."
    echo ""
    echo ""
    echo "The default connection to the target platform (HPC) login node is through OpenSSH, configured by default in ~/.ssh (see: man ssh_config)."
    echo ""
    echo "The default ports for worker peers begin at 50000 (PORTRANGE_BEGIN) and the list of PEERS is generated as:"
    echo "PEERS = ( WORKER1_IP:PORTRANGE_BEGIN, WORKER1_IP:PORTRANGE_BEGIN+1, ..., WORKER2_IP:PORTRANGE_BEGIN, WORKER2_IP:PORTRANGE_BEGIN+1, ... )"
    echo ""
    echo "Logs can be found at [pathToBuild]/logs."
    echo ""
    echo ""
    echo "Examples:"
    echo "  $0 singularity && $0 build && $0 package                Builds the Singularity image and uses it to compile the build directory codes, then packages it."
    echo "  $0 --login HPC --user hpc -i ~/.ssh/hpc.pub transfer    Transfers a package to the target platform through OpenSSH, using login node HPC, user hpc, and identify key hpc.pub."
    echo "  $0 -l HPC start                                         Using login node HPC, accesses the target platform and starts workers on remote machines."
    echo "  $0 -l HPC -n 1024 run example-time.daphne               Runs one request (script called example-time.daphne) on the deployment using 1024 cores, login node HPC, and default OpenSSH configuration."
    echo "  $0 run                                                  Executes one request (DAPHNE script input from standard input) at a running deployed platform, using default singularity/srun configurations."
    echo "  $0 deploy -n 10                                         Deploys once at the target platform through OpenSSH using default login node (localhost), then cleans."
    echo "  $0 workers -R=\"-t 120 --mem-per-cpu=10G --cpu-bind=cores --cpus-per-task=2\"  Starts workers at a running deployed platform using custom srun arguments (2 hours dual-core with 10G memory)."
    echo "  $0 run -R=\"--time=30 --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1\"  Executes a request with custom srun arguments (30 minutes single-core)."
    echo "  cat example.daphne | $0 run                             Example request job from a pipe."
}


#*****************************************************************************
# Parse arguments
#*****************************************************************************

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -h|--help)
        PrintHelp
        shift 1
        exit 0
        ;;
    -i)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_IDENTITY_FILE=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi        
        ;;
    -u|--user)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_USERNAME=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi        
        ;;
    -l|--login)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            SSH_LOGIN_NODE_HOSTNAME=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi        
        ;;
    -d|--pathToBuild)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PATH_TO_DEPLOY_BUILD=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi
        ;;
    -n|--numcores)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            NUMCORES=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi
        ;;
    -p|--port)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PORTRANGE_BEGIN=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi
        ;;
    --args)
        if [ -n "$2" ]; then
            ARGS_CS="--args $2 "
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            PrintHelp
            exit 1
        fi
        ;;
    -S=-*|--ssh-arg=-)
        SSH_ARG="${1:3}"
        shift 1
        ;;        
    -C=-*|--scp-arg=-)
        SCP_ARG="${1:3}"
        shift 1
        ;;        
    -R=-*|--srun-arg=-)
        SRUN_ARG="${1:3}"
        shift 1
        ;;        
    -G=-*|--singularity-arg=-)
        SINGULARITY_ARG="${1:3}"
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


# parse commands
PARAMS=""
while (( "$#" )); do
  case "$1" in
    singularity)
        SINGULARITY_BUILD_AND_TRANSFER_COMMAND=TRUE
        shift 1
        ;;
    build)
        BUILD_WITH_CONTAINER_COMMAND=TRUE
        shift 1
        ;;
    package)
        PACKAGE_SCRIPTS_AND_BUILD_COMMAND=TRUE
        shift 1
        ;;
    transfer)
        TRANSFER_PACKAGE_COMMAND=TRUE
        shift 1
        ;;
    start)
        START_WORKERS_COMMAND=TRUE
        shift 1
        ;;
    workers)
        START_WORKERS_SLURM_COMMAND=TRUE
        shift 1
        ;;
    status)
        WORKERS_STATUS_COMMAND=TRUE
        shift 1
        ;;
    wait)
        WAIT_WORKERS_COMMAND=TRUE
        shift 1
        ;;
    stop)
        KILL_WORKERS_COMMAND=TRUE
        shift 1
        ;;
    run)
        RUN_ONE_REQUEST_COMMAND=TRUE
        shift 1
        ;;
    clean)
        CLEAN_PLATFORM_COMMAND=TRUE
        shift 1
        ;;
    deploy)
        DEPLOY_COMMAND=TRUE
        shift 1
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
    echo "Missing: please specifcy where the build is located."
    exit 1
fi

if [[ ! -n $SSH_IDENTITY_FILE ]]; then
    echo "Missing: please configure the identity file for OpenSSH client (argument -i or inside the script)."
    exit 1
fi

# reevaluate the commands
SSH_COMMAND="ssh ${SSH_IDENTITY_FILE:+-i $SSH_IDENTITY_FILE} ${SSH_PORT:+-p $SSH_PORT} $SSH_ARG"
SCP_COMMAND="scp ${SSH_IDENTITY_FILE:+-i $SSH_IDENTITY_FILE} ${SSH_PORT:+-P $SSH_PORT} $SCP_ARG"

if [[ -n $PARAMS ]]; then
    DAPHNE_SCRIPT_AND_PARAMS="$PARAMS"
fi

if [[ -n $SINGULARITY_BUILD_AND_TRANSFER_COMMAND ]]; then   
    BuildAndTransferSingularityContainerImage
    exit 0
fi

if [[ -n $BUILD_WITH_CONTAINER_COMMAND ]]; then   
    BuildWithSingularityContainer
    exit 0
fi

if [[ -n $PACKAGE_SCRIPTS_AND_BUILD_COMMAND ]]; then   
    PackageBuiltDaphnePayload
    exit 0
fi

if [[ -n $TRANSFER_PACKAGE_COMMAND ]]; then   
    TransferPackageToTargetPlatform
    exit 0
fi

if [[ -n $START_WORKERS_COMMAND ]]; then
    RemotelyStartWorkers
    exit 0
fi

if [[ -n $START_WORKERS_SLURM_COMMAND ]]; then
    StartWorkersInContainersOnSLURM
    exit 0
fi

if [[ -n $WORKERS_STATUS_COMMAND ]]; then
    WorkersStatus
    exit 0
fi

if [[ -n $WAIT_WORKERS_COMMAND ]]; then
    WaitAllWorkersUp
    exit 0
fi

if [[ -n $KILL_WORKERS_COMMAND ]]; then
    KillWorkersOnSLURM
    exit 0
fi

if [[ -n $RUN_ONE_REQUEST_COMMAND ]]; then
    RunOneRequest
    exit 0
fi

if [[ -n $CLEAN_PLATFORM_COMMAND ]]; then
    DeploymentClean
    exit 0
fi

if [[ -n $DEPLOY_COMMAND ]]; then
    DeployEverythingHere
    exit 0
fi

PrintHelp
