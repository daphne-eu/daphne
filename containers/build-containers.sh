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

export DOCKER_BUILDKIT=1

docker build -t daphneeu/daphne-dev --build-arg NUM_CORES="$(nproc)" \
    --build-arg DAPHNE_REPO="https://github.com/daphne-eu/daphne.git" --build-arg DAPHNE_BRANCH="main" -f ./daphne-dev.Dockerfile .

docker build -t daphneeu/daphne-dev-interactive -f ./daphne-dev-interactive.Dockerfile .
