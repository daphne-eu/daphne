#!/usr/bin/env bash

# Copyright 2023 The DAPHNE Consortium
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

repetitions=1

# daphne directory inside the container
export DAPHNE_ROOT=/daphne

function timepoint () {
    date +%s%N
}

function run () {

      printf " --- DAPHNE connected components sparse base (no cuda, no codegen) $(date) \n "
      for repIdx in $(seq $repetitions)
      do
#          printf "$(date) repetition $repIdx...\n "

          local start=$(timepoint)

          $DAPHNE_ROOT/bin/daphne --timing --select-matrix-repr --config $DAPHNE_ROOT/D7.3/UserConfig.json \
              $DAPHNE_ROOT/D7.3/cc-read.daph inFile=\"$DAPHNE_ROOT/data/amazon/amazon.mtx\"  outFile=\"cc-out-base.csv\"\
              | tee $DAPHNE_ROOT/D7.3/DAPHNE-base-ccg-output-$repIdx.txt

          local end=$(timepoint)

          printf "DAPHNE\t$repIdx\t$(($end - $start))\n" >> $DAPHNE_ROOT/D7.3/base-runtime.csv

          printf "done.\n"
      done


#      printf " --- DAPHNE connected components sparse vectorized (no cuda, no codegen) $(date) \n "
#      for repIdx in $(seq $repetitions)
#      do
#          printf "$(date) repetition $repIdx...\n "
#
#          local start=$(timepoint)
#
#          $DAPHNE_ROOT/bin/daphne --vec --timing --select-matrix-repr --config $DAPHNE_ROOT/D7.3/UserConfig.json \
#              $DAPHNE_ROOT/D7.3/cc-read.daph inFile=\"$DAPHNE_ROOT/data/amazon/amazon.mtx\" outFile=\"cc-out-vec.csv\" \
#              | tee $DAPHNE_ROOT/D7.3/DAPHNE-base-vec-output-$repIdx.txt
#
#          local end=$(timepoint)
#
#          printf "DAPHNE\t$repIdx\t$(($end - $start))\n" >> $DAPHNE_ROOT/D7.3/vec-runtime.csv
#
#          printf "done.\n"
#      done


      printf " --- DAPHNE connected components sparse CUDA (no codegen) $(date) \n "
      for repIdx in $(seq $repetitions)
      do
#          printf "$(date) repetition $repIdx...\n "

          local start=$(timepoint)

          $DAPHNE_ROOT/bin/daphne --cuda --timing --select-matrix-repr --config $DAPHNE_ROOT/D7.3/UserConfig.json \
              $DAPHNE_ROOT/D7.3/cc-read.daph inFile=\"$DAPHNE_ROOT/data/amazon/amazon.mtx\"  outFile=\"cc-out-cuda.csv\" \
              | tee $DAPHNE_ROOT/D7.3/DAPHNE-base-cuda-output-$repIdx.txt

          local end=$(timepoint)

          printf "DAPHNE\t$repIdx\t$(($end - $start))\n" >> $DAPHNE_ROOT/D7.3/cuda-runtime.csv

          printf "done.\n"
      done


      printf " --- DAPHNE connected components with CUDA codegen (ccg) $(date) \n "
      for repIdx in $(seq $repetitions)
      do
#          printf "$(date) repetition $repIdx...\n "

          local start=$(timepoint)

          $DAPHNE_ROOT/bin/daphne --cuda --cuda_codegen --timing --select-matrix-repr --config $DAPHNE_ROOT/D7.3/UserConfig.json \
              $DAPHNE_ROOT/D7.3/cc-read.daph inFile=\"$DAPHNE_ROOT/data/amazon/amazon.mtx\"  outFile=\"cc-out-ccg.csv\" \
              | tee $DAPHNE_ROOT/D7.3/DAPHNE-ccg-output-$repIdx.txt

          local end=$(timepoint)

          printf "DAPHNE\t$repIdx\t$(($end - $start))\n" >> $DAPHNE_ROOT/D7.3/ccg-runtime.csv

          printf "done.\n"
      done

}

run