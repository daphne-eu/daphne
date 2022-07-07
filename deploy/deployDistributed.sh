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




#******************************************************************************
# Configuration
#******************************************************************************


PEERS=localhost:50051,localhost:50052

PORT=50000

PATH_TO_BUILD=~/DaphneDistributedWorker

#******************************************************************************
# SSH Configurations
#******************************************************************************

# TODO different configurations for each machine

identity_file=
USERNAME=
sshPort=22

#******************************************************************************
#           DO (not) EDIT BELOW THIS LINE
#******************************************************************************

SSH_COMMAND="ssh ${identity_file:+-i $identity_file} ${sshPort:+-p $sshPort}"
SCP_COMMAND="scp ${identity_file:+-i $identity_file} ${sshPort:+-P $sshPort}"


# Stop immediately if any command fails.
set -e

#******************************************************************************
# Build and deploy DistributedWorker
#******************************************************************************

function deploy () {
    PATH_TO_BUILD=$1
    ./build.sh --target DistributedWorker
    echo "Compressing Build folder"
    tar -czf DaphneWorker.tar.gz ./build


    #******************************************************************************
    # Connect to remote machines and deploy
    #******************************************************************************
    
    # In case we have one machine (same hostname) with different ports no need to send and re-deploy
    # Peers are sorted, just track two same consecutive hostnames

    LAST_MACHINE=""
    for i in "${PEERS[@]}"; do
        IFS=':' read -ra HOSTNAME_PORT <<< "$i"
        HOSTNAME=${HOSTNAME_PORT[0]}  
        PORT=${HOSTNAME_PORT[1]}
        if [[ $HOSTNAME != $LAST_MACHINE ]]; then
            echo "Sending compressed worker to " $HOSTNAME
            ($SCP_COMMAND DaphneWorker.tar.gz $USERNAME@$HOSTNAME:DaphneWorker.tar.gz) &
            LAST_MACHINE=$HOSTNAME
        fi
    done;
    wait
    # Same, in case of one same hostname no need to re-deploy
    LAST_MACHINE=""
    for i in "${PEERS[@]}"; do
        IFS=':' read -ra HOSTNAME_PORT <<< "$i"
        HOSTNAME=${HOSTNAME_PORT[0]}  
        PORT=${HOSTNAME_PORT[1]}
        if [[ $HOSTNAME != $LAST_MACHINE ]]; then
            echo "Uncompressing daphne to worker " $HOSTNAME
            ($SSH_COMMAND $USERNAME@$HOSTNAME " tar -xzf DaphneWorker.tar.gz; \
                                                mkdir -p $PATH_TO_BUILD; \
                                                mv build $PATH_TO_BUILD; \
                                                mkdir -p $PATH_TO_BUILD/logs; \
                                                rm DaphneWorker.tar.gz; \
                                                " ) &
            LAST_MACHINE=$HOSTNAME
        fi
    done;
    wait    
    rm DaphneWorker.tar.gz
}


#******************************************************************************
# Start workers on remote machines
#******************************************************************************

function StartWorkers {
    for i in "${PEERS[@]}"; do        
        IFS=':' read -ra HOSTNAME_PORT <<< "$i"
        HOSTNAME=${HOSTNAME_PORT[0]}
        PORT=${HOSTNAME_PORT[1]}
        echo "Starting remote workers at " $HOSTNAME:$PORT
        $SSH_COMMAND $USERNAME@$HOSTNAME "\
            cd $PATH_TO_BUILD; \
            mkdir -p logs;              \
            ./build/src/runtime/distributed/worker/DistributedWorker $HOSTNAME:$PORT > logs/$HOSTNAME.$PORT.out & echo \$! > logs/$HOSTNAME.$PORT.PID" &
    done;
}

#******************************************************************************
# Get worker's status
#******************************************************************************

function WorkerStatus {    
    for i in "${PEERS[@]}"; do
        IFS=':' read -ra HOSTNAME_PORT <<< "$i"
        HOSTNAME=${HOSTNAME_PORT[0]}
        PORT=${HOSTNAME_PORT[1]}
        echo "Checking remote worker $HOSTNAME:$PORT"
        $SSH_COMMAND $USERNAME@$HOSTNAME "\
            if [ -f $PATH_TO_BUILD/logs/$HOSTNAME.$PORT.PID ];\
            then \
            cd $PATH_TO_BUILD; \
                ps -f \$(cat $PATH_TO_BUILD/logs/$HOSTNAME.$PORT.PID) && echo 'Worker is UP' || echo 'Worker is DOWN';\
                :;\
            else \
                echo 'logs directory does not exist (maybe running script for the first time?).'; \
            fi" 
    done;
}

#******************************************************************************
# Kill workers
#******************************************************************************

function KillWorkers {
    for i in "${PEERS[@]}"; do
        IFS=':' read -ra HOSTNAME_PORT <<< "$i"
        HOSTNAME=${HOSTNAME_PORT[0]}
        PORT=${HOSTNAME_PORT[1]}
        # Killing distributed workers using saved PID
        echo "Killing remote workers at " $HOSTNAME
        ($SSH_COMMAND $USERNAME@$HOSTNAME "kill \$(cat $PATH_TO_BUILD/logs/$HOSTNAME.$PORT.PID)") &
    done;
}
#******************************************************************************
# Help message
#******************************************************************************

function printHelp {
    echo "Start the DAPHNE distributed worker on remote machines."    
    echo "Usage: $0 [-h|--help] [--deploy] [--pathToBuild] [-r| --run] [-s| --status] [--kill] [-peers IP[:PORT], ...]"
    echo ""    
    echo "Please remember to set DISTRIBUTED_WORKERS=IP:PORT,IP:PORT,... before running a DAPHNE script."
    echo "Logs can be found at [pathToBuild]/logs"
    echo ""
    echo "You can specify [ip:port, ...] list inside the script or pass it as argument."
    echo "Default port is 50000 but if you are running in local"
    echo "machine you need to specify ports [-peers IP[:PORT], ...]"
    echo ""
    echo "--deploy:"
    echo "This includes downloading and building all required third-party "
    echo "material (if necessary), for building the DistributedWorker."
    echo "You should only invoke it from DAPHNE's root directory"
    echo "(where this script resides)."    
    echo ""
    echo "Optional arguments:"
    echo "  -p, --port              Specify port number (default 50000)."
    echo "  -i identity_file        Specify identity file (default: default ssh private key)."
    echo "  --deploy                Compress and deploy build folder to remote machines."
    echo "  --pathToBuild           A path to deploy or where the build is already deployed (default ~/DaphneDistributedWorker can be specified in the script)."
    echo "  --peers [IP[:PORT],...] Specify (comma delimited) IP:PORT workers (default localhost:50051,localhost:50052)" 
    echo "  -r, --run               Run workers on remote machines."
    echo "  -s, --status            Get distributed workers' status."
    echo "  --kill                  Kill all distributed workers."
    echo "  -h, --help              Print this help message and exit."
}

#******************************************************************************
# Parse peers
#******************************************************************************

# Split hosts and apply default port if not specified
function parsePeers {
    IFS=',' read -ra PEERS <<< "$PEERS"
    for i in "${!PEERS[@]}"; do
        if [[ ${PEERS[$i]} == *":"* ]]; then
            PEERS[$i]="${PEERS[$i]}"
        else
            # else concat default port
            PEERS[$i]="${PEERS[$i]}:${PORT}"
            
        fi
    done

    # Keep peers sorted, easier to track duplicate hostnames (same machines)
    IFS=$'\n' PEERS=($(sort <<<"${PEERS[*]}"))
    unset IFS
}
# Parse in case of default peers
parsePeers 
#******************************************************************************
# Parse arguments
#******************************************************************************

PARAMS=""
while (( "$#" )); do
  case "$1" in
    --peers)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PEERS=$2
            parsePeers                        
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi                
        ;;
    -h|--help)
        printHelp
        shift 1
        exit 0
        ;;
    -p|--port)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            PORT=$2
            shift 2
        else
            echo "Error: Argument for $1 is missing" >&2
            printHelp
            exit 1
        fi
        ;;
    -i)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            identity_file=$2
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

if [[ ! -n $PATH_TO_BUILD ]]; then
    echo "You must specifcy where the build is located."
    exit 1
fi

if [[ ! -n $PEERS ]]; then
    echo "You must specifcy peers to deploy or run workers."
    exit 1
fi

if [[ ! -n $USERNAME ]]; then
    echo "Please configure ssh username (inside the script)."
    exit 1
fi

if [[ -n $DEPLOY_FLAG ]]; then   
    deploy $PATH_TO_BUILD
    exit 0
fi


if [[ -n $START_WORKERS_FLAG ]]; then
    StartWorkers    
    exit 0
fi

if [[ -n $WORKERS_STATUS_FLAG ]]; then
    WorkerStatus
    exit 0
fi

if [[ -n $KILL_WORKERS_FLAG ]]; then
    KillWorkers
    exit 0
fi

printHelp