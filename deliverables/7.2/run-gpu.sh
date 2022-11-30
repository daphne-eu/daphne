#!/bin/bash

c_red=$'\e[31m'
c_reset=$'\e[0m'

## executes DAPHNE compiler
#function daphne () {
#  cmd="${root}/bin/daphne ${@}"
#  echo -e "${c_green}\$> ${cmd} ${c_reset}"
#  $cmd 2>/dev/null # suppress std err output
#}

function print_info () {
  echo -e "${c_red}${@}${c_reset}"
}

libDir="lib"

# call DAPHNE compiler with example script and TPC-H data files
runtime_args="--libdir ${libDir} ./scripts/deliverables/lm.daph input=\"deliverables/7.2/lm_input_data.dbdf\""

print_info LM on CPU
bin/daphne ${runtime_args} #2>/dev/null

print_info LM vectorized CPU
OPENBLAS_NUM_THREADS=1 bin/daphne --vec ${runtime_args}# 2>/dev/null

print_info LM on CUDA
bin/daphne --cuda ${runtime_args} #2>/dev/null

# not working correctly yet
#print_info LM vectorized CUDA
#OPENBLAS_NUM_THREADS=1 bin/daphne --vec --cuda ${runtime_args} #2>/dev/null
