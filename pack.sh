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
usage: $0 --version VERSION

EOF
  exit 1
}

if [ $# -eq 0 ]; then
  exit_with_usage
fi

DAPHNE_FEATURES=("--arrow" "--cuda")
DAPHNE_VERSION=-1

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

export PACK_ROOT=daphne-$DAPHNE_VERSION-bin

source build.sh ${DAPHNE_FEATURES[@]} --target all

# shellcheck disable=SC2154
if [ -d "$daphneBuildDir"/venv ]; then
  source "$daphneBuildDir"/venv/bin/activate
else
    if ! command -v virtualenv &> /dev/null
    then
      echo "you should install virtualenv to create a python environment for testing"
      echo "e.g., apt install virtualenv";
      exit
    else
      virtualenv "$daphneBuildDir"/venv
      source "$daphneBuildDir"/venv/bin/activate
      pip install numpy
    fi
fi

# shellcheck disable=SC2154
cd "$projectRoot"
source test.sh ${DAPHNE_FEATURES[@]}

# shellcheck disable=SC2181
if [[ $? == 0 ]];then
  cd "$daphneBuildDir"
  mkdir -p $PACK_ROOT/conf
  # shellcheck disable=SC2154
  cp -a bin lib "$projectRoot"/{deploy,doc,scripts} $PACK_ROOT
  cp "$projectRoot"/UserConfig.json $PACK_ROOT/
  cp "$projectRoot"/{CONTRIBUTING.md,LICENSE.txt,README.md} $PACK_ROOT/
  tar czf $PACK_ROOT.tgz $PACK_ROOT
  cd - > /dev/null
else
  echo "test.sh did not succeed - aborting $0"
fi

set +e
