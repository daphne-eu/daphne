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

Setting up a development environment for DAPHNE can be non-trivial, especially if different platforms 
(operating systems and version etc.) are considered.
Thus, we offer pre-built environments as docker container images that can be used on Linux and Windows/WSL hosts.

This documentation will explain:
* building containers and running containers
* docker container repositories
  * for users: daphneeu/daphne
  * for developers: daphneeu/daphne-dev
* container types (docker and singularity)
* accelerator support

### Basic usage:
   ```
   # 1. Pull the DAPHNE source code.
   git clone https://github.com/daphne-eu/daphne.git
   cd daphne
   
   # 2. Pull the pre-built container image from DockerHub.   
   docker pull daphneeu/daphne

   # 3. Launch DAPHNE to run some DaphneDSL script
   containers/quickstart.sh script/examples/hello-world.daph 
   ```

## Docker images
The images we provide at the moment are 
* **daphneeu/daphne**: This image provides a pre-compiled DAPHNE executable and is geared towards small image size
  (containing only a minimal set of libraries required to run).
* **daphneeu/daphne-dev**: This image provides an environment to build DAPHNE from source. Furthermore, it contains some 
additional packages that are intended to make live easier in an interactive session and an 
entrypoint script that sets up a user inside the container to circumvent permission issues and an ssh daemon to
connect to a running container (useful for working remotely via VSCode for example).
* **daphneeu/github-action**: this is used for continuous compilation via GitHub Actions
* **daphneeu/daphne-deps**: this (unprovided) image is basically a checkpoint after the thirdparty dependencies have
been built during image creation 

## Container tags
The images mentioned above come with different tags optionally support accelerator APIs:
* **latest_BASE**: The image with this tag is the latest built for "standard" CPU operations.
* **latest_CUDA**: This image incorporates the CUDA SDK (adding ~3.5GB of required disk space)
* **< date >_< flavour >_< optional-version-string >_< OS-Version >**: This is the format of tags that indicate a daily/nightly
build. E.g., <br /> ```2023-05-12_CUDA_12.1.1-cudnn8-runtime-ubuntu20.04``` identifies a **CUDA** image of CUDA version
12.1.1 with *CUDNN 8** support that is based on Ubuntu 20.04 and was generated on the 12th of May 2023. 
* For the currently imminent release 0.2 we will have a v0.2 tag. 
* All images we provide at the moment are Ubuntu 20.04 based and run on the amd64 platform only. Developers frequently
build for other OS and platform versions themselves. Contributions on the GitHub issue tracker are welcome.

## Shell scripts for containers

[//]: # (A quick intro how the shell scripts in this directory can aid in handling the DAPHNE containers is already given)
[//]: # (in [GettingStarted.md]&#40;../doc/GettingStarted.md&#41;)

The content of the scripts is quite self-explanatory but here's a shortlisting of what can be done:
* **build-containers.sh**: Use this script to build your local Docker images. The current setup in this script is 
geared towards the DAPHNE image maintainers. So you might want to comment/remove what is not needed.
* **run-docker-example.sh**: Use this as the starting point to customize scripts to launch the DAPHNE docker images from 
your command line.
* **quickstart.sh**: A quick start script to mount the current directory and run precompiled daphne. 
* Tip: You can run any of the images with customizations disabled by providing an empty entrypoint parameter
  (e.g., ``-entrypoint=``) to the ``docker run`` command.


## Building a Docker container
To build the DAPHNE containers, use the provided [``containers/build-containers.sh``](/containers/build-containers.sh) 
script contained in this directory.
Edit the script to customize the repository and branch where DAPHNE is fetched from or to comment out one build command
(e.g., if you don't want/need the interactive container).

## Building a Singularity container
To build a Singularity container instead of Docker use the following conversion technique:
  ```bash
    #one can also use [Singularity python](https://singularityhub.github.io/singularity-cli/)
    #to convert the provided Dockerfile into Singularity recipe 
    singularity build <ImageName.sif> docker://daphneeu/daphne-dev
    
    # This command will place you in a shell in the container, your home directory and /tmp mounted. 
    singularity shell <ImageName.sif>
    Singularity> cd <your/daphne/directory>
    Singularity> ./build.sh --no-deps --installPrefix /usr/local
```
- Because the container instance works on the same folder, if one already built the system outside the container, it is 
recommended to clean all build files to avoid conflicts (`./build.sh --clean -y`)
- One may also do the commits from within the containers as normal (this holds for Singularity. With Docker images, your
home directory and therefore your .gitconfig is usually not available (but this can be mitigated with additional work 
by the user)).
- The ssh remote access is not available right away. Manual work is required at the moment (and perhaps root privileges)
to make this work. What'd be required are an sshd_config, generated host keys, manual sshd invocation and a custom port. 
- For further options refer to the Singularity documentation (e.g., --nv for CUDA devices, etc).
- These instructions should (in theory) be compatible to Apptainer but this is untested.

## Compiling DAPHNE 
The provided dev containers contain a prebuilt version of the required third party dependencies. To use them (after
creating a suitable script from the ``containers/run-docker-example.sh`` blueprints to launch the container) 
with build.sh (which by defaults tries to build the dependencies in the ``thirdparty`` subdirectory) the following 
parameters are required: ``./build.sh --no-deps --installPrefix /usr/local``

## Misc little helpers
**1. With an ssh tunnel** the ssh access feature of the dev docker container (daphneeu/daphne-dev) can conveniently be used to
work inside the container with remote development features of several popular IDEs. The scenario would be as follows:
   * Compute node in a data-center running the daphne-dev docker container. This machine shall be reachable by the IP address 
     192.168.0.123 in our example. 
   * The docker container on the compute node gets assigned the docker-internal IP address of 172.17.0.2. This address is not 
   reachable from outside of the compute node. This is the address where the sshd inside the container is listening on port 22.
   * A DAPHNE user sitting at their workstation running VSCode with Remote-SSH extension. IP address of this host is not relevant 
   but this computer needs to have a network connection to the compute node of course.
     - With the setup above the following command would provide a tunnel from the workstation into the container.
     Once the tunnel is up, the Remote-SSH plugin of VSCode can connect with the docker user/password to port 2345 on the
     localhost address of the workstation.
     - ``` ssh compute-node-user@192.168.0.123 -L 2345:172.17.0.2:22```

    
**2. The password** of a running daphneeu/daphne-dev container can be retrieved (if it's been forgotten and scrolled out of sight)
by searching the log output of the container.
   * First retrieve the container id:
    ``` bash
    $ docker ps
    CONTAINER ID   IMAGE                             COMMAND                  CREATED       STATUS       PORTS     NAMES
    99a5b6c85bbb   daphneeu/daphne-dev:latest_BASE   "/entrypoint-interac…"   2 hours ago   Up 2 hours   22/tcp    tender_mcclintock
    ```
    *  The password can subsequently be retrieved with 
    ``` bash
    $ docker logs 99a5b6c85bbb | grep password
    Use docker-username with password Docker!4556 for SSH login
    ```



[//]: # (### TODO)
[//]: # (* Rebuilding the containers automatically for latest changes)
[//]: # (* Images of released versions of DAPHNE )
