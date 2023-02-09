# Building Daphne

The DAPHNE project provides a full-fledged build script. After cloning, it does everything from dependency setup to 
generation of the executable.

### What does the build script do? (simplified)

- Download & build all code dependencies
- Build Daphne
- Clean Project

### How long does a build take?

The first run will take a while, due to long compilation times of the dependencies (~40 minutes on a 12 vcore laptop, ~10 minutes on a 128 vcore cluster node). But they only have to be compiled once (except updates).
Following builds only take a few seconds/minutes.

Contents:
 - [Usage of the build script](#1-usage-of-the-build-script)
 - [Extension of the build script](#2-extension)


--- 
## 1. Usage of the build script

This section shows the possibilities of the build script.

### Build

The default command to build the default target **daphne**.

```bash
./build.sh
```

Print the cli build help page. This also shows all the following options.

```bash
./build.sh --help
```

Build a specific **target**.

```bash
./build.sh --target "target"
```
For example the following builds the main test target.
```bash
./build.sh --target "run_tests"
```


### Clean

Clean all build directories, i.e., the daphne build dir `<project_root>/build` and the build output in 
`<project_root>/bin` and  `<project_root>/lib`

```bash
./build.sh --clean
```

Clean all downloads and extracted archive directories, i.e., `<thirdparty_dir`/download-cache, `<thirdparty_dir`/sources
and `<thirdparty_dir`/*.download.success files: 

```bash
./build.sh --cleanCache
```

Clean third party build output, i.e., `<thirdparty_dir>/installed`, `<thirdparty_dir>/build` and 
`<thirdparty_dir`/*.install.success files:

```bash
./build.sh --cleanDeps
```

Clean everything (DAPHNE build output and third party directory)

```bash
./build.sh --cleanAll
```

### Minimize long compile times of dependencies
The most time consuming part of getting DAPHNE compiled is building the third party dependencies.
To avoid this, one can either use a prebuilt container image (in combination with some parameters to the build script 
see below)) or at least build the dependencies once and subsequently point to the directory where the third party
dependencies get installed. The bulid script must be invoked with the following two parameters to achieve this:
``` ./build.sh --no-deps --installPrefix <path/to/installed/deps  ```. 

If you have built DAPHNE and **change the installPrefix directory**, it is required to clean up and build again:
``` ./build.sh --clean ```

### Options
All possible options for the build script:

| Option                 | Effect                                                                                     |
|------------------------|--------------------------------------------------------------------------------------------|
| -h, --help             | Print the help page                                                                        |
| --installPrefix <path> | Install third party dependencies in <path> (default: <project_root>/thirdparty)            | 
| --clean                | Clean DAPHNE build output (<project_root>/{bin,build,lib})                                 |
| --cleanCache           | Clean downloaded and extracted third party artifacts                                       |
| --cleanDeps            | Clean third party dependency build output and installed files                              |
| --cleanAll             | Clean DAPHNE build output and reset the third party directory to the state in the git repo |
| --target \<target>     | Build specific target                                                                      |
| -nf, --no-fancy        | Disable colorized output                                                                   |
| --no-deps              | Avoid building third party dependencies                                                    |
| -y, --yes              | Accept prompt (e.g., when executing the clean command)                                     |
| --cuda                 | Compile with support for GPU operations using the CUDA SDK                                 |
| --debug                | Compile the daphne binary with debug symbols                                               |
| --oneapi               | Compile with support for accelerated operations using the OneAPI SDK                       |
| --fpgaopencl           | Compile with support for FPGA operations using the Intel FPGA SDK or OneAPI+FPGA Add-On    |
| --arrow                | Compile with support for Apache Arrow                                                      |

## 2. Extension
### Overview over the build script
The build script is divided into segments, visualized by
```
#******************************************************************************
# Segment name
#******************************************************************************
```
Each segment should only contain functionality related to the segment name.

The following list contains a rough overview over the segments and the concrete functions or functionality done here. 
1. Help message
   1. **printHelp()** // prints help message
2. Build message helper
   1. **daphne_msg(** \<message> **)** // prints a status message in DAPHNE style
   2. **printableTimestamp(** \<timestamp> **)** // converts a unix epoch timestamp into a human readable string (e.g., 5min 20s 100ms)
   3. **printLogo()** // prints a DAPHNE logo to the console
3. Clean build directories
   1. **clean(** \<array ref dirs> \<array ref files> **)** // removes all given directories (1. parameter) and all given files (2. parameter) from disk
   2. **cleanBuildDirs()** // cleans build dirs (daphne and dependency build dirs)
   3. **cleanAll()** // cleans daphne build dir and wipes all dependencies from disk (resetting the third party directory) 
   4. **cleanDeps()** // removes third party build output
   5. **cleanCache()** // removes downloaded third party artifacts (but leaving git submodules (only LLVM/MLIR at the time of writing)
4. Create / Check Indicator-files
   1. **dependency_install_success(** \<dep> **)** // used after successful build of a dependency; creates related indicator file 
   2. **dependency_download_success(** \<dep> **)** // used after successful download of a dependency; creates related indicator file
   3. **is_dependency_installed(** \<dep> **)** // checks if dependency is already installed/built successfully
   4. **is_dependency_downloaded(** \<dep> **)** // checks if dependency is already downloaded successfully
5. Version configuration
   1. Versions of the software dependencies are configured here
6. Set some paths
   1. Definition of project related paths
   2. Configuration of path prefixes. For example all build directories are prefixed with `buildPrefix`. If fast storage 
      is available on the system, build directories could be redirected with this central configuration.
7. Parse arguments
   1. Parsing
   2. Updating git submodules
8. Download and install third-party material if necessary
   1. Antlr
   2. catch2
   3. OpenBLAS
   4. nlohmannjson
   5. abseil-cpp - Required by gRPC. Compiled separately to apply a patch.
   6. gRPC
   7. MLIR
9. Build DAPHNE target
   1. Compilation of the DAPHNE-target ('daphne' is default)

### Adding a dependency
1. Create a new subsegment in segment 8.
2. Define needed dependency variables
   1. DirName
   2. Version
   3. Dep-specific paths
   4. Dep-specific files
   5. etc.
3. Download the dependency, encased by:
    ```
    dep_dirname="<dep_name>"
    dep_version="<dep_version>"
    if ! is_dependency_downloaded "<dep_name>_v${dep_version}"; then
    
        # do your stuff here
    
        dependency_download_success "<dep_name>_v${dep_version}"
    fi
    ```
4. Install the dependency (if necessary), encased by:
    ```
    if ! is_dependency_installed "<dep_name>_v${dep_version}"; then
    
        # do your stuff here
    
        dependency_install_success "<dep_name>_v${dep_version}"
    fi
    ```
5. Define a flag for the build script if your dependency is optional or poses unnecessary 
   overhead for users (e.g., CUDA is optional as the CUDA SDK is a considerably sized package that only owners of Nvidia hardware would want to install).

   See section 7 about argument parsing. Quick guide: define a variable and its default value and add an item to the argument handling loop.

