#!/bin/bash

# Your user name on the server.
user=krzywnicka
# The IP address of the server.
ip=130.149.237.23 # so013
# The path on the server (make sure that it exists by first creating it on the server, if necessary).
path="/home/$user/daphne-$user"

 rsync -av --exclude=/.git/ --exclude=/send-to-server.sh --rsh="ssh -J $user@130.149.237.11" . "$user@$ip:$path" 
 #rsync -av --exclude='.git/' --exclude='send-to-server.sh' --include='*/' --include='*.py' --rsh="ssh -J $user@130.149.237.11" . "$user@$ip:$path"
