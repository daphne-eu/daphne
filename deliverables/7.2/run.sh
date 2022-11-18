#!/bin/bash

c_green=$'\e[32m'
c_reset=$'\e[0m'

# check realpath dependency
if [ ! $(command -v realpath) ]; then
  echo "Please install realpath (e.g. $ sudo apt-get install -y realpath)."
  exit 1
fi

# executes DAPHNE compiler
function daphne () {
  # TODO set to build directory
  cmd="${root}/bin/daphne ${@}"
  echo -e "${c_green}\$> ${cmd} ${c_reset}"
  $cmd
}

# path this script is called from
home=$(pwd)
# absolute path to this script file
script_dir="$(realpath $(dirname "${0})"))"
# DAPHNE dir
root=$(realpath "$script_dir/../..")

libDir="lib"

# TPC-H files
customer="${root}/benchmarks/tpc-h/data/customer.csv"
orders="${root}/benchmarks/tpc-h/data/orders.csv"

# navigate into DAPHNE dir
cd "$root" || ( cd "$home" && exit 1 )

echo "DAPHNE dir: ${root}"
echo "Customer data: ${customer}"
echo "Orders data: ${orders}"

# call DAPHNE compiler with example script and TPC-H data files
runtime_args="--libdir ${libDir} ./scripts/deliverables/del_7_2_example.daphne inCustomer=\"${customer}\" inOrders=\"${orders}\""
explains="--explain parsing --explain sql --explain kernels"

#daphne ${explains} "${@}" ${runtime_args}
daphne "${@}" ${runtime_args}

if ! [ -d "${script_dir}/output" ]; then
  mkdir "${script_dir}/output"
fi

echo "Still doing work, don't abort."

# save explain output
#daphne ${explains} ${runtime_args} > "${script_dir}/output/log_wo_kernellib.txt" 2> "${script_dir}/output/explains_wo_kernellib.txt"
#cat "${script_dir}/output/log_wo_kernellib.txt"
#daphne --kernellib MorphStore ${explains} ${runtime_args} > "${script_dir}/output/log_kernellib.txt" 2> "${script_dir}/output/explains_kernellib.txt"
#cat "${script_dir}/output/log_kernellib.txt"


# go back to initial directory
cd "$home" || exit 1
