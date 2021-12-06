#!/bin/bash
set -x
########################
NUMCORES=10 # change this to parameterize the number of distributed workers
#export DEMO_USE_CUDA="-p gpu"
DAPHNEparam_components_N=6000
DAPHNEparam_components_e=30
########################

tar xf build.tgz # unpack the workload

echo -e "\nSpawning N new distributed worker daemons, N=" $NUMCORES
mkdir -p WORKERS/; rm WORKERS/* 2>/dev/null # clean workerlist

srun -J Dworkers ${DEMO_USE_CUDA} --cpu-bind=cores --cpus-per-task=2 -n $NUMCORES bash -c 'singularity exec ../d.sif build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID )) > WORKERS/WORKERS.$(hostname):$(( 50000 + SLURM_LOCALID )) 2>&1' &

#until [ $(cat WORKERS.* | grep "Started Distributed Worker on " | wc -l) -ge $NUMCORES ]
date  +"Time is: "%F+%T
echo -n "\nWaiting for workers to become available ..."
set +x
until [ $(cd WORKERS; ls -1 WORKERS.* 2>/dev/null | wc -l) -ge $NUMCORES ]
do
	echo -n .
	sleep 1
done
set -x
#export WORKERS=$(cat WORKERS.* | awk '{print $NR}' | sed -e 's/Started Distributed Worker on `//g' -e 's/`$//g' | xargs -d\,)
date  +"Time is: "%F+%T

echo -e "\nThis is the demo .daphne executable that will be run:"
cat components-42-time.daphne

echo -e "\nSuccessfully spawned N new distributed worker daemons, N=" $NUMCORES
squeue -u ales.zamuda # print the generated worker list

echo -e "\n...starting the use of workers..."
sleep 5
date  +"Time is: "%F+%T

# ONE WORKER
set +x
export WORKERS=$(cd WORKERS; echo WORKERS* | sed -e 's/WORKERS.//g' -e 's/ /,/g' | sed -e 's/,.*$//g')
set -x
for DEMO_SEQUENCE in {1..5}; do
        echo -e "\n" Using ONLY ONE DISTRIBUTED WORKER $WORKERS: running the demo sequence no. $DEMO_SEQUENCE ...

	time srun ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec components-42-time.daphne --args n='${DAPHNEparam_components_N}' e='${DAPHNEparam_components_e} | awk '{a[NR]=$0} END {print(a[1](a[4]-a[2])/1000000000, "seconds with input generation, ", (a[4]-a[3])/1000000000, "seconds for compute WITH ONLY ONE DISTRIBUTED WORKER"); for (i=5; i<=NR; i++)printf("%s ",a[i]);print;}'
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

	time srun ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec components-42-time.daphne --args n='${DAPHNEparam_components_N}' e='${DAPHNEparam_components_e} | awk '{a[NR]=$0} END {print(a[1](a[4]-a[2])/1000000000, "seconds with input generation, ", (a[4]-a[3])/1000000000, "seconds for compute"); for (i=5; i<=NR; i++)printf("%s ",a[i]);print;}'
done


# TEARING DOWN
echo -e "\n\nTearing down distributed worker daemons ..."
scancel -n Dworkers


wait
exit

#time srun --cpu-bind=cores --nodes=5 --ntasks-per-node=20 --cpus-per-task=2 singularity exec ../d.sif build/src/runtime/distributed/worker/DistributedWorker
#time srun --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 singularity exec ../d.sif build/bin/daphnec sqlexample.daphne
#time srun --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 singularity exec ../d.sif build/bin/daphnec sqlexample.daphne

#srun --cpu-bind=cores --cpus-per-task=2 -n 1 bash -c 'singularity exec ../d.sif build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID ))'  
#srun --cpu-bind=cores --cpus-per-task=2 -n 1 bash -c 'singularity exec ~/sing/d.sif ~/sing/d/build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID ))' 
#export WORKERS=$(cat WORKERS.* |  xargs |sed -e 's/ /,/g')
        time singularity exec ../d.sif bash -c "DISTRIBUTED_WORKERS=localhost:50000,localhost:50001 build/bin/daphnec e.daphne --vec"
