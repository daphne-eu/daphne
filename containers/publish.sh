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

# Stop immediately if any command fails.
set -e

echo This is a maintainer script to push docker images
echo
exit

TIMESTAMP_DATE=$(date -I)

# swap comments to tag a specific version
VERSION=$TIMESTAMP_DATE
#VERSION=v0.2-rc1

# read needed software versions (e.g., CUDA version,...)
source ../software-package-versions.txt

#on some installations docker can only be run with sudo
USE_SUDO=
#USE_SUDO=sudo

ARCH=X86-64
if [ $(arch) == 'armv*'  ] || [ $(arch) == 'aarch64' ]; then
  echo "Building for ARMv8 architecture"
  ARCH=ARMV8
fi

# cuda dev image
$USE_SUDO docker tag daphneeu/daphne-dev:${TIMESTAMP_DATE}_${ARCH}_CUDA_${cudaVersion}-cudnn8-devel-ubuntu${ubuntuVersion} daphneeu/daphne-dev:${VERSION}_${ARCH}_CUDA_${cudaVersion}-cudnn8-devel-ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne-dev:${VERSION}_${ARCH}_CUDA_${cudaVersion}-cudnn8-devel-ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne-dev:latest_${ARCH}_CUDA

# base dev image
$USE_SUDO docker tag daphneeu/daphne-dev:${TIMESTAMP_DATE}_${ARCH}_BASE_ubuntu${ubuntuVersion} daphneeu/daphne-dev:${VERSION}_${ARCH}_BASE_ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne-dev:${VERSION}_${ARCH}_BASE_ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne-dev:latest_${ARCH}_BASE

# cuda run image
$USE_SUDO docker tag daphneeu/daphne:${TIMESTAMP_DATE}_${ARCH}_CUDA_${cudaVersion}-cudnn8-runtime-ubuntu${ubuntuVersion} daphneeu/daphne:${VERSION}_${ARCH}_CUDA_${cudaVersion}-cudnn8-runtime-ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne:${VERSION}_${ARCH}_CUDA_${cudaVersion}-cudnn8-runtime-ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne:latest_${ARCH}_CUDA

# base run image
$USE_SUDO docker tag daphneeu/daphne:${TIMESTAMP_DATE}_${ARCH}_BASE_ubuntu${ubuntuVersion} daphneeu/daphne:${VERSION}_${ARCH}_BASE_ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne:${VERSION}_${ARCH}_BASE_ubuntu${ubuntuVersion}
$USE_SUDO docker push daphneeu/daphne:latest_${ARCH}_BASE