#!/bin/bash
#TODO (issue ticket): write a parameters description & help printing for users of this file

echo ""
echo Parameters of the "run.sh" script: $*

########################
#export DEMO_USE_CUDA="-p gpu"
[ -z "$NUMCORES" ] && export NUMCORES=10 # change this to parameterize the number of distributed workers
[ -z "$DAPHNEparam_components_N" ] && export DAPHNEparam_components_N=6000
[ -z "$DAPHNEparam_components_e" ] && export DAPHNEparam_components_e=30
########################

echo -e "\nNUMCORES=$NUMCORES, DAPHNEparam_components_N=$DAPHNEparam_components_N, DAPHNEparam_components_e=$DAPHNEparam_components_e"

echo "These files will be unpacked:"
echo "--------------------------------------"
tar xof build.tgz 2>&1 | xargs # unpack the workload
echo "--------------------------------------"
echo ""

echo -e "\nInfo about the daphnec build/ dir is:"
cat build/git_source_status_info

echo -e "\nSpawning N new distributed worker daemons, N=" $NUMCORES
set +x
mkdir -p WORKERS/; rm WORKERS/* 2>/dev/null # clean workerlist

set -x
srun -J Dworkers --time=119 --mem-per-cpu=20G ${DEMO_USE_CUDA} --cpu-bind=cores --cpus-per-task=2 -n $NUMCORES bash -c 'singularity exec ../d.sif build/src/runtime/distributed/worker/DistributedWorker $(hostname):$(( 50000 + SLURM_LOCALID )) > WORKERS/WORKERS.$(hostname):$(( 50000 + SLURM_LOCALID )) 2>&1' &
set +x

#until [ $(cat WORKERS.* | grep "Started Distributed Worker on " | wc -l) -ge $NUMCORES ]
date  +"Time is: "%F+%T
echo -en "\nWaiting for workers to become available ..."
set +x
until [ $(cd WORKERS; ls -1 WORKERS.* 2>/dev/null | wc -l) -ge $NUMCORES ]
do
	echo -n .
	sleep 1
done
#export WORKERS=$(cat WORKERS.* | awk '{print $NR}' | sed -e 's/Started Distributed Worker on `//g' -e 's/`$//g' | xargs -d\,)
date  +"Time is: "%F+%T

export Run_Algorithm_name=components-42-time.daphne
[ -z "$3" ] || export Run_Algorithm_name=$3
echo -e "\nThis is the demo $Run_Algorithm_name executable that will be run:"
echo -e "--------------------------------------"
cat ${Run_Algorithm_name}
echo -e "\n--------------------------------------"
echo ""

echo -e "\nSuccessfully spawned N new distributed worker daemons (see queue below), N=" $NUMCORES
squeue -u ales.zamuda # print the generated worker list

echo -e "\n...starting the use of workers after 5 seconds..."
sleep 5
date  +"Time is: "%F+%T


set -x
#----------------------------- BEGIN --components_read
[ "$1" == "--components_read" ] && (
# ALL WORKERS
set +x
export WORKERS=$(cd WORKERS; echo WORKERS* | sed -e 's/WORKERS.//g' -e 's/ /,/g')
set -x
squeue -u ales.zamuda

echo "Mapping datasets..."
export DISTRIBUTED_WORKERS=$WORKERS
export COO_to_CSS_scale_factor=1
[ -z "$2" ] || export COO_to_CSS_scale_factor=$2 
export Run_Algorithm_name=components_read.daphne
[ -z "$3" ] || export Run_Algorithm_name=$3
SCRATCHfs_dir=/exa5/scratch/user/$(whoami)/

(echo "The estimated memory usage (thanks, according to model by Patrick Damme) for this run overhead is: "
w=$NUMCORES sf=$COO_to_CSS_scale_factor G="57*(2^20)" v=403394 ; bc <<< "scale=2; $sf * (2 * $G/$w  + $v*8 + $v*32/$w) / (2^30)"; echo GB ) | xargs

if [[ "$4" == "--cached-datasets" ]]; then
	# re-use the handles template and insert the new worker names
	echo "REUSING cached datasets."
	#mv datasets not-used-datasets
	rm -rf datasets
	ln -s /exa5/scratch/user/ales.zamuda/${NUMCORES}/datasets .
	echo $DISTRIBUTED_WORKERS | sed -e 's/,/\n/g' | paste '-' datasets/handles-template | sed -e 's@\t.*,datasets@,datasets@g' > datasets/Amazon0601_handles.csv
	echo "Handles (head 5, tail 5):"
	cat datasets/Amazon0601_handles.csv | head -n 10
	cat datasets/Amazon0601_handles.csv | tail -n 10
else
	echo "Generating partitioned dataset"
	set +x; rm datasets/Amazon0601* 2>/dev/null; 

#	mv datasets ${SCRATCHfs_dir}/
#	ln -s ${SCRATCHfs_dir}/datasets/ .

	date  +"Time is: "%F+%T
	set -x
	time srun --time=119 --mem-per-cpu=20G --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d2.sif python3 ./COO_to_CSV-distributed.py datasets/amazon0601.txt 403394 $NUMCORES $COO_to_CSS_scale_factor datasets/Amazon0601 COOFormat
	set +x
	cat datasets/Amazon0601_handles.csv
fi

date  +"Time is: "%F+%T

echo -e "\nReady to run this demo executable in a sequence using all distributed workers ..."

for DEMO_SEQUENCE in {1..5}; do
        echo -e "\n" Running the demo sequence no. $DEMO_SEQUENCE ...

	####
	#### RUNNING WORKLOAD ....
	####
	#printing short beginning and ending of the output:
		#echo "... only the mixed components:"
		#cat | grep "DenseMatrix(" | awk '{for (i=1; i<=NF; i++) if ($i-i+2) printf("%s ", $i-i+2); print"";}'
	set -x
	time srun --time=119 --mem-per-cpu=20G ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
		singularity exec ../d.sif bash \
			-c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec '${Run_Algorithm_name}' --args f=\"datasets/Amazon0601_handles.csv\" --select-matrix-representations' | 
		awk '{a[NR]=$0} END {print(a[2]/1000000000, "seconds for compute", a[1], a[2]); for (i=3; i<=NR; i++)printf(" %s",a[i]);print;}' | 
		( cat | grep "DenseMatrix(" | awk '{for (i=1; i<=(NF>15?15:NF); i++) printf($i" "); if (NF>15) {printf(" ... ");for (i=(NF-15)>0?NF-25:1; i<=NF; i++) printf($i" "); print""}}') 
	set +x
done

if [[ "$4" == "--cached-datasets" ]]; then
	echo "Not changing the cache."
else
	echo "Archiving the cache. (not.)"
#	mv ${SCRATCHfs_dir}/datasets ../cache/datasets-$NUMCORES-$COO_to_CSS_scale_factor #preserve the dataset
	#set +x; rm -rf ${SCRATCHfs_dir}/datasets/ 2>/dev/null; set -x
fi

# TEARING DOWN
echo -e "\n\nTearing down distributed worker daemons ..."
scancel -n Dworkers
)
[ "$1" == "--components_read" ] && exit
#----------------------------- END


# ONE WORKER
set +x
export WORKERS=$(cd WORKERS; echo WORKERS* | sed -e 's/WORKERS.//g' -e 's/ /,/g' | sed -e 's/,.*$//g')
set -x
for DEMO_SEQUENCE in {1..5}; do
        echo -e "\n" Using ONLY ONE DISTRIBUTED WORKER $WORKERS: running the demo sequence no. $DEMO_SEQUENCE ...

	time srun --time=30 ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec components-42-time.daphne --args n='${DAPHNEparam_components_N}' e='${DAPHNEparam_components_e} | awk '{a[NR]=$0} END {print((a[2]-a[1])/1000000000, "seconds for compute WITH ONLY ONE DISTRIBUTED WORKER"); for (i=3; i<=NR; i++)printf(" %s",a[i]);print;}'
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

	time srun --time=30 ${DEMO_USE_CUDA} --cpu-bind=cores --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 singularity exec ../d.sif bash -c 'DISTRIBUTED_WORKERS='${WORKERS}' build/bin/daphnec components-42-time.daphne --args n='${DAPHNEparam_components_N}' e='${DAPHNEparam_components_e} | awk '{a[NR]=$0} END {print((a[2]-a[1])/1000000000, "seconds for compute"); for (i=3; i<=NR; i++)printf(" %s",a[i]);print;}'
done


# TEARING DOWN
echo -e "\n\nTearing down distributed worker daemons ..."
scancel -n Dworkers


wait
exit

