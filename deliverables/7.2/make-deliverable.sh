#!/usr/bin/env bash

VERSION=D7.2
FINAL_DIR=daphne-$VERSION
# check realpath dependency
if [ ! $(command -v realpath) ]; then
  echo "Please install realpath (e.g. $ sudo apt-get install -y realpath)."
  exit 1
fi

# path this script is called from
home=$(pwd)
# absolute path to this script file
script_dir="$(realpath $(dirname "${0})"))"
# DAPHNE dir
root=$(realpath "$script_dir/../..")

# navigate into DAPHNE dir
cd "$root" || ( cd "$home" && exit 1 )

mkdir $FINAL_DIR

deliverables/7.2/pack-deliverable.sh --version $VERSION --feature cuda "$@"
cp build/daphne-*.tgz $FINAL_DIR

deliverables/7.2/pack-deliverable.sh --version $VERSION --feature fpgaopencl "$@"
cp build/daphne-*.tgz $FINAL_DIR

deliverables/7.2/pack-deliverable.sh --version $VERSION --feature morphstore "$@"
cp build/daphne-*.tgz $FINAL_DIR

git --no-pager log -1 > $FINAL_DIR/gitlog-HEAD.txt
git archive --prefix=daphne-$VERSION-source/ -o daphne-$VERSION-source.tgz HEAD
mv daphne-$VERSION-source.tgz $FINAL_DIR
tar czf daphne-$VERSION.tgz $FINAL_DIR
rm -rf $FINAL_DIR

# go back to initial directory
cd "$home" || exit 1