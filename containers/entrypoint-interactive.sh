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

/usr/sbin/sshd -f /etc/ssh/sshd_config


# Allow root login and password authentication
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/KbdInteractiveAuthentication no/KbdInteractiveAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/ChallengeResponseAuthentication no/ChallengeResponseAuthentication yes/' /etc/ssh/sshd_config

# Allow port forwarding
sed -i 's/#AllowTcpForwarding yes/AllowTcpForwarding yes/' /etc/ssh/sshd_config
sed -i 's/#GatewayPorts no/GatewayPorts yes/' /etc/ssh/sshd_config

#enable logging
sed -i 's/#SyslogFacility AUTH/SyslogFacility AUTH/' /etc/ssh/sshd_config
sed -i 's/#LogLevel INFO/LogLevel INFO/' /etc/ssh/sshd_config

# Uncomment the Port 22 line
sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config

echo "root:x" | chpasswd

/usr/sbin/sshd -D &

/usr/sbin/groupadd -g "$GID" dockerusers
/usr/sbin/useradd -c 'Docker Container User' -u $UID -g "$GID" -G sudo -m -s /bin/bash -d /home/"$USER" "$USER"
printf "${USER} ALL=(ALL:ALL) NOPASSWD:ALL" | sudo EDITOR="tee -a" visudo #>> /dev/null
mkdir -p /home/"$USER"/.ssh
chmod 700 /home/"$USER"/.ssh
touch /home/"$USER"/.sudo_as_admin_successful
# set a default password
SALT=$(date +%M%S)
PASS=x # Docker!"1234"
#echo "${USER}":"$PASS" | chpasswd
echo
echo For longer running containers consider running \'unminimize\' to update packages
echo and make the container more suitable for interactive use.
echo
echo "Use "$USER" with password "$PASS" for SSH login"
echo "Docker Container IP address(es):"
awk '/32 host/ { print f } {f=$2}' <<< "$(</proc/net/fib_trie)" | grep -vE "127.0." | sort -u
# shellcheck disable=SC2068
#/usr/sbin/sshd -D &
#exec "$@"

# Restart SSH service
service ssh restart

# Add rsync to PATH
export PATH=$PATH:/usr/bin
export PATH=$PATH:/usr/bin/rsync

exec su "$USER" -c $@
sudo --preserve-env=PATH,LD_LIBRARY_PATH,TERM -u $USER $@