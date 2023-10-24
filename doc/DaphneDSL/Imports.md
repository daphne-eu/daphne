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

# Imports

How to import functions from other Daphne scripts

Example usage:

```cpp
import "bar.daphne";
import "foo.daphne" as "utils";

print(bar.x);
print(utils.x);
```

---

`UserConfig.json` now has a new field `daphnedsl_import_paths`, which maps e.g., library names to a list of paths, see example:

```json
    "daphnedsl_import_paths": 
    {
        "default_dirs": ["test/api/cli/import/sandbox", "some/other/path"],
        "algorithms": ["test/api/cli/import/sandbox/algos"]
    }
```

NOTE: `default_dirs` can hold many paths and it will look for the **one** specified file in each, whereas any other library names have a list consisting of **one** directory,  from which **all** files will be imported (can be easily extended to multiple directories).

Example:

```cpp
import "a.daphne";
import "algorithms";

print(a.x);
print(algorithms.kmeans.someVar);
```

The first import will first check if the relative path exists, then it will look for it relative to paths in `default_dirs`. If the specified file exists for more than one relative path, an error will be thrown.
The second import goes to `algorithms` directory from `UserConfig` and imports all files from it.

Paths from `UserConfig` get to `DaphneDSLVisitor` from `daphne.cpp` via `DaphneUserConfig`.

---

Variable name collision resolution:
Whenever we stumble upon equal prefixes (e.g., files with the same name in different directories), a parent directory of the file where conflict is detected is prepended before prefix.

Example:

```cpp
import "somedir/a.daphne";
import "otherdir/a.daphne";

print(a.x);
print(otherdir.a.x);
```

NOTE: the parent directory may be prepended even though you never specified it (e.g., the import script is in the same directory as the original script).

Example:

```cpp
import "somedir/a.daphne";
import "a.daphne";

print(a.x);
print(otherdir.a.x);
```

Libraries and aliases:
Currently, the following example is allowed:

```cpp
import "algorithms";
import "sandbox/b.daphne" as "algorithms"; 

print(algorithms.x);
print(algorithms.kmeans1.someVar);
```

Even though both prefixes will begin with `algorithms.`, the entire library content's prefix is extended with filenames. It is up to user to not confuse yourself.

---

Cascade imports:
Any variables/functions imported into the script we are currently importing will be discarded.
Example import scheme: `A<-B<-C`. A imports B, B imports C. B uses some vars/functions from C, but A doesn't "see" any of C's content.
