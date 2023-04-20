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

/usr/sbin/groupadd -g "$GID" dockerusers
/usr/sbin/useradd -c 'Docker Container User' -u $UID -g "$GID" -G sudo -m -s /bin/bash -d /home/"$USER" "$USER"
printf "${USER} ALL=(ALL:ALL) NOPASSWD:ALL" | sudo EDITOR="tee -a" visudo #>> /dev/null
touch /home/"$USER"/.sudo_as_admin_successful
exec su "$USER"
# set a default password
#echo ${USER}:Docker! | chpasswd
