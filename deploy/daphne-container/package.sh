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

# This script packages the package of built dode and the user code (scripts)
# that should be run when deployed with DAPHNE.
# The step of packaging using same compilation (build_*) is hence reusable.

echo "Packaging latest files for daphnec/DistributedWorker deployment..."

cd container

(
tar cvzf build.tgz build/
tar cvzf packet.tgz build.tgz e.daphne run.sh
) | awk '{printf("\r%-100s      ", substr($0, -1, 100));}'

