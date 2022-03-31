algo=components_read.daphne

./store-one-deployment-output.sh -r 10 1
./store-one-deployment-output.sh -r 10 10 $algo --cached-datasets 
./store-one-deployment-output.sh -r 10 100 $algo --cached-datasets 
./store-one-deployment-output.sh -r 10 300 $algo --cached-datasets 
./store-one-deployment-output.sh -r 100 1 
./store-one-deployment-output.sh -r 100 100 $algo --cached-datasets
./store-one-deployment-output.sh -r 100 300 $algo --cached-datasets 
./store-one-deployment-output.sh -r 300 1
./store-one-deployment-output.sh -r 300 300 $algo --cached-datasets
