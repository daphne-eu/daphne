#!/usr/bin/env bash

docker run -e USER=$(id -un) -e UID=$(id -u) -e GROUP=$(id -gn) -e GID=$(id -g) --rm -it --gpus all \
     -v $PWD:/workdir -w /workdir --entrypoint=/workdir/entrypoint.sh daphne-ubi8 bash
