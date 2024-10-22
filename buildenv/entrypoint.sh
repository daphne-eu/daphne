#!/usr/bin/env bash

echo Creating user: $USER
groupadd -r sudo 
groupadd -g $GID $GROUP
useradd -u $UID -g $GID -G sudo,users -m $USER
dnf install -y sudo 
printf "${USER} ALL=(ALL:ALL) NOPASSWD:ALL" | sudo EDITOR="tee -a" visudo > /dev/null
su $USER
