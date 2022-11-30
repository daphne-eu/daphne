#!/usr/bin/env bash

# if this is run from the source snapshot, ask to recompile
if [[ -d src ]]; then
  if [[ -d bin || -d build || -d lib ]]; then
    while true; do
    read -p "bin, build or lib directory exist. Cleanup first to rebuild? (y/n) " yn
    case $yn in
      y ) echo cleaning;
        rm -rf bin build lib
        break;;
      n ) echo skipping cleanup;
        break;;
      * ) echo invalid response;;
    esac
    done
  fi

  if ! [[ -d bin || -d build || -d lib ]]; then
    ./build.sh --cuda
  else
    echo skipping rebuild
  fi
fi

ROWS=500000
COLS=1000
VAL_SIZE=4 # fp32 = 4 bytes
let DATASIZE=$ROWS*$COLS*$VAL_SIZE/1024**2
echo Creating test input file lm_input_data.dbdf with $ROWS rows and $COLS columns \(size=${DATASIZE}Mb\)
bin/daphne scripts/deliverables/lm_datagen.daph numRows=500000 numCols=1000 filename=\"deliverables/7.2/lm_input_data.dbdf\"
