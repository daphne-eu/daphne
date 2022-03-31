#!/bin/bash

# Copyright 2021 The DAPHNE Consortium
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

# This script builds DAPHNE code and puts it in a time-stamped build
# directory. The build is done using the DAPHNE singularity container
# "d.sif" created from the DAPHNE Docker image, at daphne-container/sing/.
# The daphnec and DistributedWorker are both built.

time singularity exec d.sif ./build.sh

time singularity exec d.sif ./build.sh --target DistributedWorker

TIME_BUILT=$(date  +%F-%T)

mv build build_${TIME_BUILT}

