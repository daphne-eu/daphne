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

# Profiling DAPHNE using PAPI

You can profile your DAPHNE script by using the ```--enable-profiling``` CLI
switch.

DAPHNE supports profiling via the [PAPI](https://github.com/icl-utk-edu/papi)
profiling library, specifically the
[high-level (HL) PAPI API](https://github.com/icl-utk-edu/papi/wiki/PAPI-HL).

When run with profiling enabled, the DAPHNE compiler will generate code that
automatically starts and stops profiling (via PAPI) at the start and end of the
DAPHNE script.

You can configure which events to profile via the `PAPI_EVENTS`
environmental variable, e.g.:

```bash
$ PAPI_EVENTS="perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE MISSES,perf::BRANCHES,perf::BRANCH-MI SSES" PAPI_REPORT=1 ./daphne --enable-profiling script.daph
```

For more details about the supported events as well as other PAPI-HL configuration
options you can check the
[PAPI HL API documentation](https://github.com/icl-utk-edu/papi/wiki/PAPI-HL#overview-of-environment-variables).
You can also get a list of the supported events on your machine via the
`papi_native_avail` PAPI utility (included in the `papi-tools` package
on Debian-based systems).
