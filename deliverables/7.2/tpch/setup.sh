#!/bin/bash

# check realpath dependency
if [ ! $(command -v realpath) ]; then
  echo "Please install realpath (e.g. $ sudo apt-get install -y realpath)."
  exit 1
fi

# path this script is called from
home=$(pwd)
# absolute path to this script file
script_dir="$(realpath $(dirname "${0})"))"


function setup_dbgen () {
  if [ -d dbgen ] && [ -f ./dbgen/dbgen ]; then return; fi
  echo "Setup generator."
  rm -rf dbgen
  # download
  git clone https://github.com/electrum/tpch-dbgen.git dbgen
  # build
  cd dbgen || exit 1
  make -j
  # exit
  cd "$script_dir" || exit 1
}

function generate () {
  if [ ! -d data ]; then
    mkdir data;
    echo "Generate data."
  fi
  cd dbgen || exit 1
  if [ ! -f ../data/orders.tbl ]; then
    ./dbgen -f -s 0.001 -T o
    mv ./orders.tbl ../data/
  fi
  if [ ! -f ../data/customer.tbl ]; then
    ./dbgen -f -s 0.001 -T c
    mv ./customer.tbl ../data/
  fi
  cd ..
}

function preprocess () {
  if [ ! $(command -v python3) ]; then
    echo "python3 not found, please install"
    exit 1
  fi
  if ! python3 -c "import pandas"; then
    python3 -m pip install pandas || (echo "Could not install pandas module. Please install manually." && exit 1)
  fi
  if [ ! -f ./data/customer.csv ] || [ ! -f ./data/orders.csv ]; then
    echo "Preprocess data."
    python3 preprocess.py data
  fi
}

function main () {
  cd "$script_dir" || exit 1
  setup_dbgen
  generate
  preprocess
  cd "$home" || exit 1
}

main
