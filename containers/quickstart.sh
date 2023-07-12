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

#on some installations docker can only be run with sudo
USE_SUDO=
#USE_SUDO=sudo

# user info to set up your user inside the container (to avoid creating files that then belong to the root user)
USERNAME=$(id -n -u)
GID=$(id -g)
$USE_SUDO docker run --rm -w /daphne -v $PWD:/daphne -e GID=$GID -e USER=$USERNAME -e UID=$UID daphneeu/daphne:latest_X86-64_BASE $*