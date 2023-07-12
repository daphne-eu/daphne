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

The DAPHNE runtime system is designed with the goal of supporting various distributed runtime that relies on various technologies, e.g. MPI and RPC.

This document shows how a DAPHNE user can execute DAPHNE scripts on a distributed computing environment with the MPI backend implementation of the DAPHNE runtime system.
This document assumes that the DAPHNE was build with the `--mpi` options, if this is not the case please rebuild DAPHNE with the `--mpi` option
```./build.sh --mpi```

The DAPHNE build script uses [Open MPI](https://www.open-mpi.org/).
The DAPHNE build script does not configure the Open MPI installation with the SLURM support option.
For users who want to add the SLURM, please visit the [Open MPI](https://www.open-mpi.org/) documentation (adding ```--with-slurm``` to the build command of the Open MPI libbrary) and edit the DAPHNE build script.
Also, users who wants to use other MPI implementations e.g., Intel MPI may edit the corresponding part in the DAPHNE build script.

## When DAPHNE is Installed Natively (w/o Container)

1. Ensure that your system knows about the installed MPI  
    -- The ```PATH``` and ```LD_LIBRARY_PATH```environment variable has to be updated as follows  

    ```bash
    export PATH=$PATH:<DAPHNE_INSTALLATION>/thirdparty/installed/bin/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<DAPHNE_INSTALLATION>//thirdparty/installed/lib/ 
    ```

    Please do not forget to replace `<DAPHNE_INSTALLATION>` with the actual path

1. Run basic example @ ```/examples/matrix_addition_for_mpi.daph``` as follows

    ```bash
    mpirun -np 10 ./bin/daphne --distributed --dist_backend=MPI scripts/examples/matrix_addition_for_mpi.daph
    ```

The command above executes 10 processes **locally** on one machine.

In order to run on **a distributed system**, you need to provide the machine names or the machinefile which contains the machine names.
For instance assuming that ```my_hostfile``` is a text file that contains machine names

```bash
mpirun -np 10 --hostfile my_hostfile  ./bin/daphne --distributed --dist_backend=MPI scripts/examples/matrix_addition_for_mpi.daph
```

The command above starts 10 processes distributed on following the hosts in the my_hostfile.
For more options, please check the [Open MPI documentation](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile).

From a DAPHNE runtime point of view, the ```--distributed``` option tells the DAPHNE runtime system to utilize the distributed backend, while the ```--dist_backend=MPI```
indicate the type of the backend implementation.

## When DAPHNE is Installed with Containers (e.g. singularity)

The main difference is that the mpirun command is called at the level of the container as follows

```bash
mpirun -np 10 singularity exec <singularity-image> daphne/bin/daphne --distributed   --dist_backend=MPI --vec --num-threads=2 daphne/scripts/examples/matrix_addition_for_mpi.daph
```

Please do not forget to replace `<singularity-image>` with the actual singularity image.
