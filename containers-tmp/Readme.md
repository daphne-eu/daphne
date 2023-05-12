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

# Getting Started with DAPHNE in a Docker Container

Setting up a development environment for DAPHNE can be non-trivial, especially if different platforms (operating systems and version etc.) are considered.
Thus, we offer pre-built environments as docker container images that can be used on Linux and Windows/WSL hosts.
We are currently refactoring the containers to make them easier to use in various situations, but we will still need some time to finish that.
In the meantime, please use the following work-around to set up your DAPHNE development environment in a container.
You can view these instructions as a complete alternative to GettingStarted.md.
Note that downloading the container image and building DAPHNE for the first time may take some time (depending on your internet connection and system setup, up to a few hours). But any subsequent builds will be much faster.
Furthermore, before you start, make sure you have installed git and docker.

1. Pull the DAPHNE source code.
   ```
   git clone --recursive https://github.com/daphne-eu/daphne.git
   ```
1. Pull the pre-built container image from DockerHub.
   ```
   docker pull daphneeu/daphne-dev
   ```
1. Change into the base path of the DAPHNE source tree.
   ```
   cd daphne
   ```
1. Create a new directory and save two auxiliary script files there.
   ```
   mkdir containers-tmp
   wget https://raw.githubusercontent.com/daphne-eu/daphne/containers-tmp/containers-tmp/run-docker.sh -O containers-tmp/run-docker.sh
   wget https://raw.githubusercontent.com/daphne-eu/daphne/containers-tmp/containers-tmp/entrypoint.sh -O containers-tmp/entrypoint.sh
   chmod u+x containers-tmp/run-docker.sh containers-tmp/entrypoint.sh
   ```
1. Start the DAPHNE docker container. You must do this from the base path of the DAPHNE source tree.
   ```
   containers-tmp/run-docker.sh
   ```
1. Now you should be inside the container, where everything is set up to use DAPHNE. You should see something like:
   ```
   <yourname> ALL=(ALL:ALL) NOPASSWD:ALL
   Use <yourname> with password Docker!1234
   <yourname>@daphne-container:/daphne$ 
   ```
1. Inside the container, you can:
   - Build DAPHNE as usual by `./build.sh`.
   - Run the test cases as usual by `./test.sh`.
   - Run DAPHNE with some DaphneDSL script as usual by `bin/daphne scripts/examples/hello-world.daph`.
1. To leave the container, execute the command `exit`.

The DAPHNE source tree on your host machine is mounted inside the container, i.e., the container directly works on the same files as your host system.
This has the advantage that you can easily use your favorite IDE on your host system to edit the DAPHNE source code.
Any changes you make will be visible inside the container, where you can simply rebuild and rerun DAPHNE with your changes.
Likewise, you can use git on the host system.
As an alternative, some simple editors as well as git are also available inside the container.