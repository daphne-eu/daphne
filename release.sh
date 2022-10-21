#!/usr/bin/env bash

# Copyright 2022 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stop immediately if any command fails.
set -e

function exit_with_usage {
  cat << EOF
usage: $0 --version VERSION --githash GIT_HASH [ --gpgkey GPG_KEY ] [ --artifact PATH_TO_FILE ] [ --feature FEATURE ]

--gpgkey: The key ID to use for signing. If supplied, an attempt to sign the artifact will be made.
          Consider setting GNUPGHOME to point to your GPG keyring.

--artifact: If supplied, building the release artifact will be skipped and the script will only perform
            checksumming and optional signing.
--feature FEATURE......a feature flag like --cuda, --arrow, etc (omit or "none" for plain Daphne)
EOF
  exit 1
}

if [ $# -eq 0 ]; then
  exit_with_usage
fi

# by default this script does not inform Github about the release (safety measure)
DRY_RUN=1

DAPHNE_VERSION=-1
GIT_HASH=0
BUILD=1
GPG_KEY=0
ARTIFACT_PATH=""
DAPHNE_REPO_URL="git@github.com:daphne-eu/daphne.git"

FEATURE="--feature "
while [[ $# -gt 0 ]]; do
    key=$1
    shift
    case $key in
        -h|--help)
            exit_with_usage
            ;;
        --version)
            DAPHNE_VERSION=$1
            shift
            ;;
        --feature)
            FEATURE=$FEATURE$1
            shift
            ;;
        --githash)
            GIT_HASH=$1
            shift
            ;;
        --gpgkey)
            GPG_KEY=$1
            shift
            ;;
        --artifact)
            ARTIFACT_PATH=$1
            BUILD=0
            shift
            ;;
        --no-dryrun)
            echo "no-dryrun selected! Release will be tagged on Github($DAPHNE_REPO_URL)"
            echo "You have 10 seconds to abort (press Ctrl-c)"
            sleep 10
            DRY_RUN=0
          ;;
        *)
            unknown_options="${unknown_options} ${key}"
            ;;
    esac
done

if [ -n "$unknown_options" ]; then
  printf "Unknown option(s): '%s'\n\n" "$unknown_options"
  exit_with_usage
fi

if [[ "$DAPHNE_VERSION" == "-1" ]] || [[ $GIT_HASH == "0" ]]; then
  echo Error: version and git hash have to be specified
  exit_with_usage
fi

if [ $BUILD -eq 1 ]; then
  if [[ "$FEATURE" == "--feature " ]]; then
    FEATURE=
  fi

  echo "Building Daphne $DAPHNE_VERSION release from git commit $GIT_HASH"

  # set the requested commit to build Daphne
  git checkout "$GIT_HASH"
  # shellcheck disable=SC2086
  source pack.sh --version "$DAPHNE_VERSION" $FEATURE
  # return to the previous branch
  git checkout -

  mkdir -p artifacts
  cd  artifacts
  echo "$GIT_HASH" > "$PACK_ROOT.githash"
  # shellcheck disable=SC2154
  mv "$daphneBuildDir/$PACK_ROOT.tgz" .
  sha512sum "$PACK_ROOT.tgz" > "$PACK_ROOT.tgz.sha512sum"
  sha512sum -c "$PACK_ROOT.tgz.sha512sum"
  echo
  cd - > /dev/null

  if ! [[ "$GPG_KEY" -eq "0" ]]; then
    gpg --detach-sign --armor --default-key "$GPG_KEY" "$PACK_ROOT".tgz
    if ! [[ "$DRY_RUN" -eq "1" ]]; then
      git tag -a -u "$GPG_KEY" "$DAPHNE_VERSION" "$GIT_HASH"
      git push "$DAPHNE_REPO_URL" --tags
    fi
    echo
    echo "Now upload artifacts to Github in the web form for creating releases"
  else
    echo "No GPG Key given - don't forget to tag and sign the artifact manually"
    echo
    echo "git tag -a -u <GPG_KEY> $DAPHNE_VERSION $GIT_HASH"
    echo "git push $DAPHNE_REPO_URL --tags"
  fi
else
  sha512sum "$ARTIFACT_PATH" > "$ARTIFACT_PATH.sha512sum"
  sha512sum -c "$ARTIFACT_PATH.sha512sum"
  echo

  echo "$GPG_KEY"

  if [ "$GPG_KEY" != "0" ]; then
    gpg --detach-sign --armor --default-key "$GPG_KEY" "$ARTIFACT_PATH"
  else
    echo "No GPG Key given - don't forget to sign the artifact manually"
    GPG_KEY="<GPG_KEY>"
  fi

  echo "Now go to your git working copy of Daphne source and issue these commands (GPG key needed) before completing"
  echo "the release in the Github web interface."
  echo
  echo "git tag -a -u $GPG_KEY $DAPHNE_VERSION $GIT_HASH"
  echo "git push $DAPHNE_REPO_URL --tags"
fi

set +e
