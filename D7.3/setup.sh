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

echo "Fetching DAPHNE container"
#docker pull daphneeu/daphne-dev:latest_X86-64_CUDA

export DAPHNE_ROOT=$PWD/daphne

echo "Deflating amazon co-purchasing dataset"
zstd -f -d $DAPHNE_ROOT/data/amazon/amazon.mtx.zst

echo "Preparing SSB data"
cd $DAPHNE_ROOT/D7.3/data_generation/
./data_gen.sh -sf 1
cd -
