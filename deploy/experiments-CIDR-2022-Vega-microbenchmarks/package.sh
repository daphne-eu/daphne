#!/bin/bash
echo "Packaging latest files for daphnec/DistributedWorker deployment..."

(
tar cvzf build.tgz build/
tar cvzf packet.tgz build.tgz *.daphne run*.sh datasets/ COO_to_CSV-distributed.py
) | awk '{printf("\r%-100s      ", substr($0, -1, 100));}'

