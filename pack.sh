#!/usr/bin/env bash

# Copyright 2022 The DAPHNE Consortium
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

# Stop immediately if any command fails.
set -e

function exit_with_usage {
  cat << EOF
usage: pack.sh --version VERSION --feature FEATURE
--feature FEATURE......a feature flag like --cuda, etc (omit or "none" for plain Daphne)
EOF
  exit 1
}

if [ $# -eq 0 ]; then
  exit_with_usage
fi

DAPHNE_VERSION=-1
FEATURE=

while [[ $# -gt 0 ]]; do
    key=$1
    shift
    case $key in
        -h|--help)
            exit_with_usage
            ;;
        --version)
            DAPHNE_VERSION=$1
            shift
            ;;
        --feature)
            # shellcheck disable=SC2001
            FEATURE=$(echo "$1" | sed 's/ *$//g')
            shift
            ;;
        *)
            unknown_options="${unknown_options} ${key}"
            ;;
    esac
done

if [ -n "$unknown_options" ]; then
  printf "Unknown option(s): '%s'\n\n" "$unknown_options"
  exit_with_usage
fi

if [[ "$DAPHNE_VERSION" == "-1" ]]; then
  echo Error: version not supplied
  exit_with_usage
fi

# shellcheck disable=SC2254
case "$FEATURE" in
  cuda)  ;&
  debug) ;&
  fpgaopencl)
    echo "Using feature $FEATURE"
    FEATURE="--$FEATURE"
    ;;
  (none) ;&
  ("")
    echo "Building plain Daphne"
    FEATURE=
    ;;
  (*)
    echo "Warning: Unsupported feature $FEATURE ignored!"
    sleep 3
    FEATURE=
    ;;
esac

ARCH=X86-64
if [ $(arch) == 'armv*'  ] || [ $(arch) == 'aarch64' ]; then
  echo "Building for ARMv8 architecture"
  ARCH=ARMV8
fi
PACK_ROOT1=daphne$FEATURE-$ARCH-$DAPHNE_VERSION-bin
#minor cosmetics replacing -- with -
export PACK_ROOT="${PACK_ROOT1/--/-}"

echo "Directories bin, build and lib will be removed before compiling."
read -p "Are you sure? [y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "\nDid not receive a \"y\". \tAborting.\n"
    git checkout -
    exit 1
fi
rm -rf bin build lib

# shellcheck disable=SC2086
# MPI is an internal feature so we can turn this on here
source build.sh -nd -ns -nf --installPrefix /usr/local --mpi $FEATURE --target all


# this might be obsolete when running from daphne-dev docker container:
## shellcheck disable=SC2154
#if [ -d "$daphneBuildDir"/venv ]; then
#  source "$daphneBuildDir"/venv/bin/activate
#else
#    if ! command -v virtualenv &> /dev/null
#    then
#      echo "you should install virtualenv to create a python environment for testing"
#      echo "e.g., apt install virtualenv";
#      exit
#    else
#      virtualenv "$daphneBuildDir"/venv
#      source "$daphneBuildDir"/venv/bin/activate
#      pip install numpy
#    fi
#fi

# shellcheck disable=SC2154
cd "$projectRoot"

# shellcheck disable=SC2086
source test.sh --no-build $FEATURE

# shellcheck disable=SC2181
if [[ $? == 0 ]];then
  cd "$daphneBuildDir"
  mkdir -p "$PACK_ROOT/bin"
  # shellcheck disable=SC2154
  cp -a "$projectRoot"/{containers,deploy,doc,lib,scripts} "$PACK_ROOT"
  mkdir -p "$PACK_ROOT"/{bin,src/api}
  cp -a "$projectRoot"/src/api/python "$PACK_ROOT"/src/api/
  cp -a "$projectRoot"/bin/{daphne,DistributedWorker} "$PACK_ROOT"/bin/
  cp -a "$projectRoot"/run-*.sh "$PACK_ROOT"/
  # this assumes that the pack script is run from an environment that has third party deps in /usr/local
  # e.g. the daphne-dev docker container
  cp -a /usr/local/lib/lib*.so* "$PACK_ROOT/lib"
  cp "$projectRoot"/{CITATION,CONTRIBUTING.md,KEYS.txt,LICENSE.txt,README.md,UserConfig.json} "$PACK_ROOT"
  tar czf "$PACK_ROOT".tgz "$PACK_ROOT"
  cd - > /dev/null
else
  echo "test.sh did not succeed - aborting $0"
fi

set +e
