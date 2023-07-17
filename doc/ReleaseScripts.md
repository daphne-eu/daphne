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

# Release Scripts

How to use the release scripts to create binary artifacts

This is a quick write-up of how the scripts to create a binary release artifact are meant to be used.

The release.sh script will call pack.sh which will call build.sh and test.sh. Only if testing completes successfully, the artifact, a gzipped tar archive (format open for discussion) is created. The command after ``--githash`` fetches the git hash of the current commit. The script checks out this git hash and restores the current commit after successful completion. This is a bit of a shortcoming as you have to issue a manual ``git checkout -`` if the script fails and terminates early.

## Signing

The release manager will have to sign the artifacts to verify that the provided software has been created by that person.
To create an appropriate GPG key, [these instructions](https://downloads.apache.org/systemds/KEYS) can be adapted to our needs. The
keys of Daphne release managers will be provided in [this file](/KEYS.txt). Ideally, future release managers sign each others keys. Key signing is a form of showing
that the one key owner trusts the other.

## The Procedure (Preliminary for v0.1)

1. Get into a bash shell and change to your working copy (aka daphne root) directory.
1. **Create the artifacts (plain Daphne):** ``./release.sh --version 0.1 --githash `git rev-parse HEAD` ``
1. **Create additional artifacts with extra features compiled in:**<br /> ``./release.sh --version 0.1 --githash `git rev-parse HEAD` --feature cuda``
<br />Note that this adds additional constraints on the binaries (e.g., if CUDA support is compiled in, the executable will fail to load on a system without the CUDA SDK properly installed)_
1. **Copy the artifacts** to a machine where you have your top secret signing key installed (can be skipped if this is the build machine):<br />
   ``rsync -vuPah <hostname>:path/to/daphne/artifacts .``
1. **Signing and checksumming:**

    ``` bash
    cd artifacts
    ~/path/to/daphne/release.sh --version 0.1 --artifact ./daphne-0.1-bin.tgz --gpgkey <GPG_KEY_ID> --githash `cat daphne-0.1-bin.githash` 
    ```

   * repeat for other feature artifacts
1. **Tag & push** The previous signing command will provide you with two more git commands to tag the commit that the artfiacts were made from and to push these tags to github.
    This should look something like this:

    ``` bash
    git tag -a -u B28F8F4D 0.1 312b2b50b4e60b3c5157c3365ec38383d35e28d8
    git push git@github.com:corepointer/daphne.git --tags
    ```

1. **Upload & release**:
    * Click the "create new release" link on the front page of the Daphne github repository (right column under "Releases").
    * Select the tag for the release, create a title, add release notes (highlights of this release, list of contributors, maybe a detailed change log at the end)
    * Upload the artifacts: All the ``<filename>.{tgz,tgz.asc,tgz.sha512sum}`` files before either saving as draft for further polishing or finally release the new version.
