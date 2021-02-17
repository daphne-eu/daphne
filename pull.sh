#!/bin/bash

# This script pulls upstream changes to this repository and handles all updates
# to the submodule(s).

git pull && git submodule update --init --recursive