<!--
Copyright 2023 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## DAPHNE container images

This directory contains scripts and config files to build and run containers 
that can be used to build and run DAPHNE. At the moment we support Docker and Singularity.

The two containers we provide at the moment are 
* **daphne-dev**: Use this to provide a container environment to the build script:<br/>
``container/run-daphne-dev.sh ./build.sh`` <br/>
This container is also used by our GitHub Action to test-build contributions to the main branch (and pull requests to main). 
* **daphne-dev-interactive**: This container contains some additional convenience packages that are intended to
make live easier in an interactive session. Additionally, it contains an entrypoint script that sets up 
a user inside the container to circumvent permission issues.

### Shell scripts for containers
A quick intro how the shell scripts in this directory can aid in handling the DAPHNE containers is already given
in [GettingStarted.md](../doc/GettingStarted.md)

The content of the scripts is quite self-explanatory but here's a shortlisting of what can be done:
* **build-containers.sh**: Use this script to build your local Docker images. Alternatively, they can also be pulled 
from Docker Hub: ``docker pull daphneeu/daphne-dev-interactive``
* **run-daphne-dev-interactive.sh**: This script properly starts the daphne-dev-interactive container
* **run-daphne-dev.sh**: Starts the non-interactive container, sets UID for permissions and mounts the current directory.

### Building a container
To build the DAPHNE containers, use the provided ``build-containers.sh`` script contained in this directory.
Edit the script to customize the repository and branch where DAPHNE is fetched from or to comment out one build command
(e.g., if you don't want/need the interactive container).

### Compiling DAPHNE 
The provided containers contain a prebuilt version of the required third party dependencies. To use them 
with build.sh (which by defaults tries to build the dependencies in the ``thirdparty`` subdirectory) the following parameters are required:
``./build.sh --no-deps --installPrefix /usr/local``

### TODO
* **daphneeu/daphne** container with a prebuilt DAPHNE executable
* CUDA and OneAPI images
* Rebuilding the containers automatically for latest changes
* Images of released versions of DAPHNE 
