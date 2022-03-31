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

# This script generates a package for deployment of the built daphne
# code, to be sent to Vega, executed, and then cleaned.
# For transfer, scp is used, since rsync is already used for the d.sif.

cd container

# this requires you to have the proper ssh config for the user of scp
scp packet.tgz login.vega.izum.si:sing/d

# likewise, config the ssh access
sshvega <<EOF
cd sing/d
tar xvf packet.tgz
./run.sh
rm -rf ~/sing/d/*
exit
EOF


