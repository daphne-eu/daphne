<!--
Copyright 2024 The DAPHNE Consortium

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

# Building in unsupported environments

DAPHNE can also be built in environments other than the one endorsed by the DAPHNE project team. To help with that,
there are scripts to build your own toolchain in the [``buildenv``](/buildenv) directory.
These scripts can be used to build the compiler and necessary libraries to build DAPHNE in unsupported environments 
(RHEL UBI, CentOS, ...) Besides the build scripts, there is a docker file to create a UBI image to build for Redhat 8.


Usage:
* Create Docker image with build-ubi8.sh
* Run the Docker image with run-ubi8.sh
* Run build-all.sh
* Run source env.sh inside or outside the Docker image to set PATH and linker variables to use the created environment 
(you need to cd into the directory containing env.sh as this uses $PWD)


Beware that this procedure needs ~50GB of free disk space. Also, the provided CUDA SDK expects a recent driver (550+) 
version. That will most likely be an issue on large installations - exchange the relevant version and file names for 
another CUDA version in env.sh to build another version.