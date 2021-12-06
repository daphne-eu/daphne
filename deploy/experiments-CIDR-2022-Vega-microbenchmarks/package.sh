#!/bin/bash
#cp -p ~/MOEA/projects/codes/daphne/*.daphne .
#cp -p ~/sing/d/e.daphne .

echo "Packaging latest files for daphnec/DistributedWorker deployment..."

(
tar cvzf build.tgz build/
tar cvzf packet.tgz build.tgz *.daphne run.sh
) | awk '{printf("\r%-100s      ", substr($0, -1, 100));}'

