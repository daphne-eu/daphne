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

# This script runs a distributed workload. It spawns new workers using 
# SLURM's srun.

set -x
########################
NUMCORES=1000 # change this to parameterize the number of distributed workers
#export DEMO_USE_CUDA="-p gpu"
########################

tar xf build.tgz # unpack the workload

echo -e "\nSpawning N new distributed worker daemons, N=" $NUMCORES
mkdir -p WORKERS/; rm WORKERS/* 2>/dev/null # clean workerlist

srun -J Dworkers ${DEMO_USE_CUDA} --cpu-bind=cores --cpus-per-task=2 -n $NUMCORES bash -c 'singularity exec ../d.sif build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID )) > WORKERS/WORKERS.$(hostname):$(( 50000 + SLURM_LOCALID )) 2>&1' &

#until [ $(cat WORKERS.* | grep "Started Distributed Worker on " | wc -l) -ge $NUMCORES ]
echo -n "\nWaiting for workers to become available ..."
set +x
until [ $(cd WORKERS; ls -1 WORKERS.* 2>/dev/null | wc -l) -ge $NUMCORES ]
do
	echo -n .
	sleep 1
done
set -x
#export WORKERS=$(cat WORKERS.* | awk '{print $NR}' | sed -e 's/Started Distributed Worker on `//g' -e 's/`$//g' | xargs -d\,)

echo -e "\nThis is the demo .daphne executable that will be run:"
cat e.daphne

echo -e "\nSuccessfully spawned N new distributed worker daemons, N=" $NUMCORES
squeue -u ales.zamuda # print the generated worker list

echo -e "\n...starting the use of workers..."
sleep 5


# ONE WORKER
set +x
export WORKERS=$(cd WORKERS; echo WORKERS* | sed -e 's/WORKERS.//g' -e 's/ /,/g' | sed -e 's/,.*$//g')
set -x
for DEMO_SEQUENCE in {1..5}; do
        echo -e "\n" Using ONLY ONE DISTRIBUTED WORKER $WORKERS: running the demo sequence no. $DEMO_SEQUENCE ...

	time srun ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec e.daphne' | awk '{a[NR]=$0} END {print(a[1](a[4]-a[2])/1000000000, "seconds with input generation, ", (a[4]-a[3])/1000000000, "seconds for compute WITH ONLY ONE DISTRIBUTED WORKER"); for (i=5; i<=NR; i++)printf("%s ",a[i]);print;}'
        sleep 1
done


# ALL WORKERS
set +x
export WORKERS=$(cd WORKERS; echo WORKERS* | sed -e 's/WORKERS.//g' -e 's/ /,/g')
set -x
squeue -u ales.zamuda

echo -e "\nReady to run this demo executable in a sequence using all distributed workers ..."

for DEMO_SEQUENCE in {1..5}; do
        echo -e "\n" Running the demo sequence no. $DEMO_SEQUENCE ...

	time srun ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec e.daphne' | awk '{a[NR]=$0} END {print(a[1](a[4]-a[2])/1000000000, "seconds with input generation, ", (a[4]-a[3])/1000000000, "seconds for compute"); for (i=5; i<=NR; i++)printf("%s ",a[i]);print;}'
done


# TEARING DOWN
echo -e "\n\nTearing down distributed worker daemons ..."
scancel -n Dworkers


wait
exit

