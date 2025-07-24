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

# HDFS Usage

This document shows how a DAPHNE user can execute DaphneDSL scripts using HDFS as a file system,
which is optimized for performance on big data through distributed computing.
This document assumes that DAPHNE was built with the `--hdfs` option, i.e., by `./build.sh --hdfs`.

DAPHNE uses [HAWQ (libhdfs3)](https://github.com/apache/hawq/archive/refs/tags/rel/v3.0.0.0.tar.gz).

## Configuring DAPHNE for HDFS

In order for DAPHNE to utilize the HDFS file system, certain command line arguments need to be passed
(or included in the configuration file).

- `--enable-hdfs`: the flag to enable hdfs
- `--hdfs-address=<IP:PORT>`: the IP and port HDFS listens to
- `--hdfs-username=<username>`: the username used to connect to HDFS

## Reading from HDFS

In order to read a file from HDFS, some preprocessing must be done. Assuming the
file is named `FILE_NAME`, a user needs to:

1. Upload the file into HDFS. DAPHNE expects the file to be located inside a directory with some specific naming conventions.
    The path can be any path under HDFS, however the file must be named with the following convention:

    ```text
    /path/to/hdfs/file/FILE_NAME.FILE_TYPE/FILE_NAME.FILE_TYPE_segment_1
    ```

    `FILE_TYPE` is either `.csv` or `.dbdf` (DAPHNE binary data format) followed by `.hdfs`, e.g. `myfile.csv.hdfs`.

    The suffix `_segment_1` is necessary, since we support multiple writers at once (see below), the writers need to write into different files (different segments).
    When the user pre-uploads the file, it needs to be in the same format, but just one segment.

    Each segment must also have its own `.meta` file in HDFS. This is a JSON
    file containing information about the size of the segment as well as the type.
    For example `myfile.csv.hdfs_segment_1.meta`:

    ```json
    {
        "numCols": 10,
        "numRows": 10,
        "valueType": "f64"
    }
    ```

2. We also need to create a `.meta` file containing information about the file, within the local file system (from where DAPHNE is invoked).
    Similar to any other file which will be read by DAPHNE, we need to create a `.meta` file, which is in JSON format, containing information about where the
    file is, information about the rows/columns etc. The file should be named `FILE_NAME.FILE_TYPE.meta`, e.g.,
    `myfile.csv.hdfs.meta`. The meta file should contain all the regular information any DAPHNE meta file contains, but in addition, it also contains information about whether this is an HDFS file and where it is located within HDFS:

    ```json
    {
        "hdfs": {
            "HDFSFilename": "/path/to/hdfs/file/FILE_NAME.FILE_TYPE",
            "isHDFS": true
        },
        "numCols": 10,
        "numRows": 10,
        "valueType": "f64"
    }
    ```

### Example

Let's say we have a dataset called `training_data.csv` which we want to upload to HDFS and use it with DAPHNE.

1. Upload file under path `datasets` and create the segment `.meta` file. HDFS should look like this:

    ```bash
    $ hdfs dfs -ls /
    /datasets/training_data.csv.hdfs/training_data.csv.hdfs_segment_1
    /datasets/training_data.csv.hdfs/training_data.csv.hdfs_segment_1.meta

    $ hdfs dfs -cat /datasets/training_data.csv.hdfs/training_data.csv.hdfs_segment_1.meta
    {"numCols":10,"numRows":10,"valueType":"f64"}
    ```

2. Create the local file `.meta` file:

    ```bash
    $ cat ./training_data.csv.hdfs.meta
    {"hdfs":{"HDFSFilename":"/datasets/training_data.csv.hdfs","isHDFS":true},"numCols":10,"numRows":10,"valueType":"f64"}
    ```

3. DAPHNE script:

    ```r
    X = readMatrix("training_data.csv.hdfs");
    print(X);
    ```

4. Run DAPHNE

    ```bash
    ./bin/daphne --enable-hdfs --hdfs-ip=<IP:PORT> --hdfs-username=ubuntu code.daph
    ```

## Writing to HDFS

In order to write to HDFS we just need to use DaphneDSL's `writeMatrix()` built-in function like we would for any other file type and specify the `.hdfs` suffix. For example:

1. Code

    ```r
    X = rand(10, 10, 0.0, 1.0, 1.0, 1);
    writeMatrix(X, "randomSet.csv.hdfs");
    ```

2. Call `daphne`

    ```bash
    ./bin/daphne --enable-hdfs --hdfs-ip=<IP:PORT> --hdfs-username=ubuntu code.daph
    ```

This will create the following files inside HDFS:

```bash
$ hdfs dfs -ls /
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1.meta

$ hdfs dfs -cat /randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1.meta
{"numCols":10,"numRows":10,"valueType":"f64"}
```

And also the `.meta` file within the local file system named `randomSet.csv.hdfs.meta`:

```json
{
    "hdfs": {
        "HDFSFilename": "/randomSet.csv.hdfs",
        "isHDFS": true
    },
    "numCols": 10,
    "numRows": 10,
    "valueType": "f64"
}
```

### Limitations

For now, writing to a specific directory, through DAPHNE, within HDFS is not supported. DAPHNE will always try to write under the root HDFS directory `/<name>.<type>.hdfs`.

## Distributed Runtime

Both read and write operations are supported by the distributed runtime.

### Read

Exactly the same preprocessing must be done, creating one file inside the HDFS with the
appropriate naming conventions. Users can then run DAPHNE using the
[distributed runtime](DistributedRuntime.md) and depending on the generated pipeline, DAPHNE's distributed workers will read their
corresponding part of the data speeding up IO significantly. For example:

1. DAPHNE script:

    ```r
    X = readMatrix("training_data.csv.hdfs");
    print(X+X);
    ```

2. Run DAPHNE

    ```bash
    $ export DISTRIBUTED_WORKERS=worker-1:<PORT>:worker-2:<PORT>
    $ ./bin/daphne --distributed --dist_backend=sync-gRPC --enable-hdfs --hdfs-ip=<IP:PORT> --hdfs-username=ubuntu code.daph
    ```

### Write

Similar to read, nothing really changes, users just need to call DAPHNE using the distributed runtime flags. Notice that since we have multiple workers/writers, more than
one segments are generated inside HDFS:

1. Code

    ```r
    X = rand(10, 10, 0.0, 1.0, 1.0, 1);
    writeMatrix(X, "randomSet.csv.hdfs");
    ```

2. Call `daphne`

    ```bash
    $ export DISTRIBUTED_WORKERS=worker-1:<PORT>:worker-2:<PORT>
    $ ./bin/daphne --distributed --dist_backend=sync-gRPC --enable-hdfs --hdfs-ip=<IP:PORT> --hdfs-username=ubuntu code.daph
    ```

Assuming two distributed workers:

```bash
$ hdfs dfs -ls /
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1        # First part of the matrix
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1.meta
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_2        # Second part of the matrix
/randomSet.csv.hdfs/randomSet.csv.hdfs_segment_2.meta

$ hdfs dfs -cat /randomSet.csv.hdfs/randomSet.csv.hdfs_segment_1.meta
{"numCols":10,"numRows":5,"valueType":"f64"}
$ hdfs dfs -cat /randomSet.csv.hdfs/randomSet.csv.hdfs_segment_2.meta
{"numCols":10,"numRows":5,"valueType":"f64"}
```

And also the `.meta` file within the local file system named `randomSet.csv.hdfs.meta`.

### Notes

It does not matter how many segments are generated or exist. DAPHNE is designed to read
the segments according to the current state (distributed or not and how many distributed
workers are being used).

For example if we use four distributed workers to write a matrix,
DAPHNE will generate four different segments. DAPHNE can later read the same matrix either in
local execution (no distributed runtime) or using a different number of workers, not depending on the amount of segments generated earlier.
