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

# Profiling DAPHNE Using PAPI

You can profile specific regions of your DaphneDSL script by inserting
`startProfiling` and `stopProfiling` calls with a string region name.
Note that profiling is only available if DAPHNE was built *without* the `--no-papi` flag.

DAPHNE supports profiling via the [PAPI](https://github.com/icl-utk-edu/papi)
profiling library, specifically the
[high-level (HL) PAPI API](https://github.com/icl-utk-edu/papi/wiki/PAPI-HL).

For example, you can mark a region like this:

```r
startProfiling("train");
# ... computation ...
stopProfiling("train");
```
The region name must match between the start and stop calls.

If you pass `--enable-profiling` and do not insert explicit profiling calls,
the compiler will wrap the entire script in a default region named `script`.

You can configure which events to profile via the `PAPI_EVENTS`
environment variable, e.g.:

```bash
$ PAPI_EVENTS="perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE MISSES,perf::BRANCHES,perf::BRANCH-MI SSES" PAPI_REPORT=1 bin/daphne script.daph
```

For more details about the supported events as well as other PAPI-HL configuration
options you can read the
[PAPI HL API documentation](https://github.com/icl-utk-edu/papi/wiki/PAPI-HL#overview-of-environment-variables).
You can also get a list of the supported events on your machine via the
`papi_native_avail` PAPI utility (included in the `papi-tools` package
on Debian-based systems).
