<!--
Copyright 2021 The DAPHNE Consortium

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

# DAPHNE Scheduling

This document describes the use of the pipeline and task scheduling mechanisms currently supported in the DAPHNE system.

## Scheduling Decisions

The DAPHNE system considers four types of scheduling decisions: work partitioning, assignment, ordering and timing.

- Work partitioning refers to the partitioning of the work into units of work (or tasks) according to a certain granularity (fine or coarse, equal or variable).
- Work assignment refers to mapping (or placing) the units of work (or tasks) onto individual units of execution (processes or threads).
- Work ordering refers to the order in which the tasks are executed. We rely on the vectorized execution engine, therefore, tasks within a vectorized pipeline have no dependencies and can be executed in any order.
- Work timing refers to the times at which the units of work are set to begin execution on the assigned units of execution.
  
**Work Partitioning**: The the DAPHNE prototype supports twelve partitioning schemes: Static (STATIC), Self-scheduling (SS), Guided self-scheduling (GSS), Trapezoid self-scheduling (TSS), Trapezoid Factoring self-scheduling (TFSS), Fixed-increase self-scheduling (FISS), Variable-increase self-scheduling (VISS), Performance loop-based self-scheduling (PLS), Modified version of Static (MSTATIC), Modified version of fixed size chunk self-scheduling (MFSC), and Probabilistic self-scheduling (PSS). The granularity of the tasks generated and scheduled by the DAPHNE system follows one of these partitioning schemes (See Section 4.1.1.1 in Deliverable 5.1 [D5.1].

**Work Assignment**: The current snapshot of the DAPHNE prototype supports two main assignment mechanisms: Single centralized work queue and Multiple work queues.  When work assignment relies on a centralized work queue (CENTRALIZED), workers follow the self-scheduling principle, i.e., whenever a worker is free and idle, it obtains a task from a central queue. When work assignment relies on multiple work queues, workers follow the work-stealing principle, i.e., whenever workers are free, idle, and have no tasks in their queues, they steal tasks from the work queue of each other. Work queues can be per worker (PERCPU) or per group of workers (PERGROUP).  In work-stealing, workers need to apply a victim selection mechanism to find a queue and steal work from it. The currently supported victim selection mechanisms are SEQ (steal from the next adjacent worker), SEQPRI (Steal from the next adjacent worker, but prioritize same NUMA domain), RANDOM (Steal from a random worker), RANDOMPRI (Steal from a random worker, but prioritize same NUMA domain).

## Scheduling Options

To list all possible execution options of the DAPHNE system, one needs to execute the following

```shell
$ ./bin/daphne --help
```

The output of this command shows all DAPHNE compilation and execution parameters including the scheduling options that are currently support. The output below shows only the scheduling options that we will cover in this document.

```shell
> This program compiles and executes a DaphneDSL script.
USAGE: daphne [options] script [arguments]
OPTIONS:
Advanced Scheduling Knobs:
  Choose task partitioning scheme:
      --STATIC             - Static (default)
      --SS                 - Self-scheduling
      --GSS                - Guided self-scheduling
      --TSS                - Trapezoid self-scheduling
      --FAC2               - Factoring self-scheduling
      --TFSS               - Trapezoid Factoring self-scheduling
      --FISS               - Fixed-increase self-scheduling
      --VISS               - Variable-increase self-scheduling
      --PLS                - Performance loop-based self-scheduling
      --MSTATIC            - Modified version of Static, i.e., instead of n/p, it uses n/(4*p) where n is number of tasks and p is number of threads
      --MFSC               - Modified version of fixed size chunk self-scheduling, i.e., MFSC does not require profiling information as FSC
      --PSS                - Probabilistic self-scheduling
  Choose queue setup scheme:
      --CENTRALIZED        - One queue (default)
      --PERGROUP           - One queue per CPU group
      --PERCPU             - One queue per CPU core
  Choose work stealing victim selection logic:
      --SEQ                - Steal from next adjacent worker
      --SEQPRI             - Steal from next adjacent worker, prioritize same NUMA domain
      --RANDOM             - Steal from random worker
      --RANDOMPRI          - Steal from random worker, prioritize same NUMA domain
  --debug-mt            - Prints debug information about the Multithreading Wrapper
  --grain-size=<int>    - Define the minimum grain size of a task (default is 1)
  --hyperthreading      - Utilize multiple logical CPUs located on the same physical CPU
  --num-threads=<int>   - Define the number of the CPU threads used by the vectorized execution engine (default is equal to the number of physcial cores on the target node that executes the code)
  --pin-workers         - Pin workers to CPU cores
  --pre-partition       - Partition rows into the number of queues before applying scheduling technique
  --vec                 - Enable vectorized execution engine
DAPHNE Options:
  --args=<string>       - Alternative way of specifying arguments to the DaphneDSL script; must be a comma-separated list of name-value-pairs, e.g., `--args x=1,y=2.2`
  --config=<filename>   - A JSON file that contains the DAPHNE configuration
  --cuda                - Use CUDA
  --distributed         - Enable distributed runtime
  --explain=<value>     - Show DaphneIR after certain compiler passes (separate multiple values by comma, the order is irrelevant)
    =parsing            -   Show DaphneIR after parsing
    =parsing_simplified -   Show DaphneIR after parsing and some simplifications
    =sql                -   Show DaphneIR after SQL parsing
    =property_inference -   Show DaphneIR after property inference
    =vectorized         -   Show DaphneIR after vectorization
    =obj_ref_mgnt       -   Show DaphneIR after managing object references
    =kernels            -   Show DaphneIR after kernel lowering
    =llvm               -   Show DaphneIR after llvm lowering
  --libdir=<string>     - The directory containing kernel libraries
  --no-obj-ref-mgnt     - Switch off garbage collection by not managing data objects' reference counters
  --select-matrix-repr  - Automatically choose physical matrix representations (e.g., dense/sparse)
Generic Options:
  --help                - Display available options (--help-hidden for more)
  --help-list           - Display list of available options (--help-list-hidden for more)
  --version             - Display the version of this program
EXAMPLES:
 daphne example.daphne
  daphne --vec example.daphne x=1 y=2.2 z="foo"
  daphne --vec --args x=1,y=2.2,z="foo" example.daphne
```

**_NOTE:_**
the DAPHNE system relies on the vectorized (tile) execution engine to support parallelism at the node level. The vectorized execution engine takes decision concerning work partition and assignment during applications’ execution. Therefore, one needs always to use the option --vec with any of the scheduling options that we present in this document.

### Multithreading Options

- **Number of threads**: A DAPHNE user can control the total number of threads spawn by the DAPHNE runtime system use the following parameter **--num-threads**. This parameter should be non-zero positive value. Illegal integer values will be ignored by the system and the default value will be used.  The default value of --num-threads is equal to the total number of physical cores of the host machine. The option can be used as below, e.g., the DAPHNE system spawns only 4 threads.

    ```shell
    ./bin/daphne --vec --num-threads=4 some_daphne_script.daphne
    ```

- **Thread Pinning**: A DAPHNE user can decide if the DAPHNE system pins its threads to the physical cores. Currently, the DAPHNE system supports one simple pining strategy, namely, round-robin strategy.  By default, the DAPHNE system does not pin its threads. The option **--pin-workers** can be used to activate thread pinning as follows

    ```shell
    ./bin/daphne --vec --pin-threads some_daphne_script.daphne
    ```

- **Hyperthreading**: if a host machine supports hyperthreading, a DAPHNE user can decide to use logical cores, i.e., if the –num-threads is not specified, the DAPHNE system sets the total number of threads to the count of the physical cores. However, when the user specify the following parameter **--hyperthreading**, the DAPHNE system sets the number of threads to the count of the logical cores.

    ```shell
    ./bin/daphne --vec --hyperthreading some_daphne_script.daphne
    ```

### Work Partitioning Options

- **Partition Scheme**: A DAPHNE user selects the partition scheme by passing the name of the partition scheme as an argument to the DAPHNE system. If the user does not specify a partition scheme, the default partition scheme (STATIC) will be used. As an example, the following command uses GSS as a partition scheme.

    ```shell
    ./bin/daphne --vec --GSS some_daphne_script.daphne
    ```

- **Task granularity**: The DAPHNE user can exploit the **--grain-size** parameter to set the minimum size of the tasks generated by the DAPHNE system. This parameter should be non-zero positive value. Illegal integer values will be ignored by the system and the default value will be used.  The default value of **--grain-size** is 1, i.e., the data associated with a task represents 1 row of the input matrix.
As an example, the following command uses SS as a partition scheme with minimum task size of 100

    ```shell
    ./bin/daphne --vec --SS --grain-size=100 some_daphne_script.daphne
    ```

### Work Assignment Options

- **Single centralized work queue**: By default, the DAPHNE system uses a single centralized work queue. However, the user may explicitly use the following parameter **--CENTRALIZED** to ensure the use of single centralized work queue.

    ```shell
    ./bin/daphne --vec --GSS --CENTRALIZED some_daphne_script.daphne
    ```

- **Multiple work queues**: a DAPHNE user can exploit the use of multiple work queues by passing one of the following parameters **--PERCPU** or **--PERGROUP**. The two parameters cannot be used together, and if **--CENTRALIZED** is used with any of them, --CENTRALIZED will be ignored by the system.
- parameter **--PERGROUP** ensures that the DAPHNE system creates a number of groups equals to the number of NUMA domains on the target host machine. The DAPHNE system assigns equal number of workers (threads) to each of the groups. Workers within the same group share one work queue. The **–-PERGROUP** can be used as follows

    ```shell
    ./bin/daphne --vec --PERGROUP some_daphne_script.daphne
    ```

- The parameter **--PERCPU** ensures that the DAPHNE system creates a number of queues equal to the total number of workers (threads), i.e., each worker is assigned to a single work queue. The parameter **--PERCPU** can be used as follows

    ```shell
    ./bin/daphne --vec --PERCPU some_daphne_script.daphne
    ```

- **Victim Selection**: A DAPHNE user can choose a victim selection strategy by passing one of the following parameters --SEQ, --SEQPRI, --RANDOM, and --RANDOMPRI. These parameters activate different victim selection strategies as follows
    - **--SEQ** activates a sequential victim selection strategy, i.e., the ith worker steals form the (i+1)th  worker. The last worker steals from the first worker.
    - **--SEQPRI** is similar to --SEQ except that --SEQPRI priorities workers assigned to the same NUMA domain. When the host machine has one NUMA domain,
    - --SEQ and --SEQPRI have no difference.
    - **--RANDOM** activates a random victim selection strategy, i.e., the ith worker steals form a randomly chosen worker.
    - **--RANDOMPRI** is similar to --RANDOM except that --RANDOM priorities workers assigned to the same NUMA domain. When the host machine has one NUMA domain, --RANDOMPRI and --RANDOMPRI have no difference.
  
**_NOTE:_**  
When the user does not choose one of these parameters, the DAPHNE system considers --SEQ as a default victim selection strategy.

As an example, the following command uses --SEQPRI as a victim selection strategy.

```shell
./bin/daphne --vec --PERGROUP --SEQPRI some_daphne_script.daphne
```

## References

[D4.1](https://daphne-eu.eu/wp-content/uploads/2021/11/Deliverable-4.1-fin.pdf) DAPHNE: D4.1 DSL Runtime Design, 11/2021

[D5.1](https://daphne-eu.eu/wp-content/uploads/2021/11/Deliverable-5.1-fin.pdf) DAPHNE: D5.1 Scheduler Design for Pipelines and Tasks, 11/2021
