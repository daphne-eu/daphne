# DAPHNE Deliverable D7.3
## Prototype and overview code generation framework

This archive contains the necessary source code, precompiled binaries and documentation to run the DAPHNE demonstartor 
in the examples described in the delivered PDF.

## Prerequisites

It is recommended to run these examples in the "daphne-dev" Docker container provided by the project via Docker Hub [1].
To run Docker containers with GPU/CUDA support, it is required to install Docker as described in [2] and the Nvidia Container
Runtime as described in [3].

Alternatively, the expert user can compile and run from the provided DAPHNE sources if a supported environment as 
described in our [getting started documentation](../doc/GettingStarted.md). 

## Folder Structure

- daphne: Subdirectory containing a snapshot of the DAPHNE repository from branch D7.3 at githash  
  - D7.3: all relevant scripts and data for running the examples of deliverable D7.3
    - data_generation Folder: Contains all the files necessary to generate the needed SSB data (with included data_gen.sh Script)
    - ssb-data Folder: Contains folder structure of data needed for benchmarks. Contains Meta files for SF1 and SF10 to load needed tables into DAPHNE
- setup.sh: Script to pull the daphneeu/daphne-dev Docker container and prepare the input data for the examples
- run.sh: Run SIMD and GPU example
- run-simd.sh: Run the SIMD example
- run-gpu.sh: Run the GPU example


[1] https://hub.docker.com/u/daphneeu.

[2] https://docs.docker.com/engine/install/#server

[3] https://docs.docker.com/config/containers/resource_constraints/#gpu