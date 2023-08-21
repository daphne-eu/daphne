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

# Logging

## General

To write out messages of any kind from DAPHNE internals we use the [spdlog](https://github.com/gabime/spdlog/) library.
E.g., not from a user's print() statement but when ``std::cout << "my value: " << value << std::endl;`` is needed. With
spdlog, the previous std::cout example would read like this: ```spdlog::info("my value: {}", value);```. The only
difference being that we now need to choose a log level (which is arbitrarily chosen to be *info* in this case).

## Usage

1. Before using the logging functionality, the loggers need to be created and registered. Due to the nature of how
singletons work in C++, this has to be done once per binary (e.g., daphne, run_tests, libAllKernels.so, libCUDAKernels.so, etc).
For the mentioned binaries this has already been taken care of (either somewhere near the main program entrypoint or
via context creation in case of the libs). All setup is handled by the class DaphneLogger (with some extras in ConfigParser).
1. Log messages can be submitted in two forms:
    - ``spdlog::warn("my warning");``
    - ``spdlog::get("default")->warn("my warning");``

    The two statements have the same effect. But while the former is a short form for using the default logger, the latter
    explicitly chooses the logger via the static ``get()`` method. This ``get()`` method is to be used with **caution** as 
    it involves acquiring a lock, which is to be avoided **in performance critical sections** of the code. In the initial 
    implementation there is a logger for runtime kernels provided by the context object to work around this limitation.
    See the matrix multiplication kernel in ``src/runtime/local/kernels/MatMult.cpp`` for example usage. 

1. We can have several loggers, which can be configured differently. For example, to control how messages are logged
in the CUDA compiler pass ``MarkCUDAOpsPass``, a logger named "compiler::cuda" is used. Additionally, avoiding the use of
``spdlog::get()`` is demonstrated there. For each used logger, an entry
in ``fallback_loggers`` (see DaphneLogger.cpp) must exist to prevent crashing when using an unconfigured logger.
1. To configure log levels, formatting and output options, the DaphneUserConfig and ConfigParser have been extended.
See an example of this in the ``UserConfig.json`` in the root directory of the DAPHNE code base.
1. At the moment, the output options of our logging infrastructure are a bit limited (initial version). A logger currently
always emits messages to the console's std-out and optionally to a file if a file name is given in the config.
1. The format of log messages can be customized. See the examples in ``UserConfig.json`` and the
[spdlog documentation](https://github.com/gabime/spdlog/).
1. If a logger is called while running unit tests (run_tests executable), make sure to ```#include <run_tests.h>``` and
call ```auto dctx = setupContextAndLogger();``` somewhere before calling the kernel to be tested.
1. Logging can be set to only work from a certain log level and above. This mechanism also serves as a global toggle.
To set the log level limit, set ```{ "log-level-limit": "OFF" },```. In this example, taken from ``UserConfig.json``,
all logging is switched off, regardless of configuration.

## Log Levels

These are the available log levels (taken from ```<spdlog/common.h>```). Since it's an enum, their numeric value
start from 0 for TRACE to 6 for OFF.

```cpp
namespace level {
enum level_enum : int
{
    trace = SPDLOG_LEVEL_TRACE,
    debug = SPDLOG_LEVEL_DEBUG,
    info = SPDLOG_LEVEL_INFO,
    warn = SPDLOG_LEVEL_WARN,
    err = SPDLOG_LEVEL_ERROR,
    critical = SPDLOG_LEVEL_CRITICAL,
    off = SPDLOG_LEVEL_OFF,
    n_levels
};
```

## ToDo

* Guideline when which log level is recommended
* Toggle console output
* Other log sinks
