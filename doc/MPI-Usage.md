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

# MPI Usage

About employing MPI as a distributed runtime backend.

The DAPHNE runtime is designed with the goal of supporting various distributed backends that rely on various technologies, e.g. MPI and RPC.

This document shows how a DAPHNE user can execute DaphneDSL scripts on a distributed computing environment with the MPI backend of the DAPHNE runtime.
This document assumes that DAPHNE was built with the `--mpi` option, i.e., by `./build.sh --mpi`.

DAPHNE's build script uses [Open MPI](https://www.open-mpi.org/).
It does not configure the Open MPI installation with the Slurm support option.
For users who want to add Slurm, please visit the [Open MPI](https://www.open-mpi.org/) documentation (adding `--with-slurm` to the build command of the Open MPI library) and edit the DAPHNE build script.
Also, users who want to use other MPI implementations, e.g., Intel MPI may edit the corresponding part in the DAPHNE build script.

## When DAPHNE is Installed Natively (w/o Container)

1. Ensure that your system knows about the installed MPI: The `PATH` and `LD_LIBRARY_PATH`environment variables have to be updated as follows:

    ```bash
    export PATH=$PATH:<DAPHNE_INSTALLATION>/thirdparty/installed/bin/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<DAPHNE_INSTALLATION>//thirdparty/installed/lib/ 
    ```

    Please do not forget to replace `<DAPHNE_INSTALLATION>` with the actual path.

1. Run the basic example `scripts/examples/matrix_addition_for_mpi.daph` as follows:
    <!-- TODO This file does not exist in the repo. By what script could we replace it? -->

    ```bash
    mpirun -np 10 ./bin/daphne --distributed --dist_backend=MPI scripts/examples/matrix_addition_for_mpi.daph
    ```

The command above executes 10 processes **locally** on one machine.

In order to run on **a distributed system**, you need to provide the machine names or the file which contains the machine names.
For instance, assuming that `my_hostfile` is a text file that contains machine names, execute the following command:

```bash
mpirun -np 10 --hostfile my_hostfile ./bin/daphne --distributed --dist_backend=MPI scripts/examples/matrix_addition_for_mpi.daph
```

The command above starts 10 processes distributed on the hosts specified in the file `my_hostfile`.
For more options, please see the [Open MPI documentation](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile).

From a DAPHNE runtime point of view, the `--distributed` option tells the DAPHNE runtime to utilize the distributed backend, while the option `--dist_backend=MPI`
indicates the type of the backend implementation.

## When DAPHNE is Installed with Containers (e.g. Singularity)

The main difference is that the `mpirun` command is called at the level of the container as follows:

```bash
mpirun -np 10 singularity exec <singularity-image> daphne/bin/daphne --distributed --dist_backend=MPI --vec --num-threads=2 daphne/scripts/examples/matrix_addition_for_mpi.daph
```

Please do not forget to replace `<singularity-image>` with the actual Singularity image.
