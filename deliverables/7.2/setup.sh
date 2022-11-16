#!/bin/bash

### path this script is called from
home=$(pwd)
### absolute path to this script file
script_dir="$(realpath $(dirname "${0})"))"
### DAPHNE dir
root=$(realpath "$script_dir/../..")




cd ${root} || ( cd ${home} && exit 1 )

### Build DAPHNE
./build.sh --morphstore

### Generate TPC-H data
./deliverables/7.2/tpch/setup.sh

cd ${home} || exit 1

