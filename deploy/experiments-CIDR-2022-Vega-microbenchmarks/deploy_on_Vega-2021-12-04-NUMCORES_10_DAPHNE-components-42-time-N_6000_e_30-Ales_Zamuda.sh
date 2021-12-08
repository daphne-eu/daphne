#!/bin/bash

# Copyright 2021 The DAPHNE Consortium
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

# This script runs an experiment on Vega.

reset;

echo "Welcome to the DAPHNE on Vega for components-42-time (DAPHNE Consortium, 2021-12-06)."
cat <<EOF
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMW0xOXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMNOddddokNMMMMMMMMMMMKxxxxOXWMMMMMMMXxkMMMMMMOxxxxkKWMMMKxXMMMMNx0MM0xKMMMMXxKMMNxxxxxxxOMMM
MMMMMMMMMXkddddd0WMMMMMMMMMMMl cxdc'.oWMMMMW,  dMMMMM' lxxl..oMMl dMMMM0 :MM: .OMMMd lMM0 ,xxxxxOMMM
MMMMM0ll0WWN0kKWWNk',OWMMMMMMl 0MMMMO :MMMMo O; KMMMM' KMMMW. KMl dMMMM0 :MM:.l kMMd lMM0 lMMMMMMMMM
MMMKo,,,;oKMMMMMO.    .OMMMMMl 0MMMMM, 0MM0 cMX.;MMMM' KMMXl ,WMl '::::, :MM:'Md.0Md lMM0 .:::::OMMM
MMWk:,,,,ckNWWWWo.    .xMMMMMl 0MMMMM, KMW' ;oo. kMMM' ,;;;ckWMMl :OOOOo :MM:.MMl.Od lMM0 ;OOOOOXMMM
MMMMWk::OWMWNWMMMNx.,kWMMMMMMl 0MMMWx cMMo cxxxx,.KMM' KMMMMMMMMl dMMMM0 :MM:.MMWc : lMM0 lMMMMMMMMM
MMMMMMWWMWNNMMMMMMMMMMMMMMMMMl cxoc..dWMK.cMMMMM0 'WM' KMMMMMMMMl dMMMM0 :MM:.MMMWl  lMMO ,xxxxxOMMM
MMMMMMMMMXXWMMMMMMMMMMMMMMMMMKxxxk0NMMMMKxNMMMMMM0xNMOxNMMMMMMMMKxXMMMMNx0MM0OMMMMMOxKMMNxxxxxxxOMMM
MMMMMMMMMMMNXWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWOod0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMX,    0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNxl;,'',,:0MMMMMX    KMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMK:  ,c,   ;XMMMMMMx   dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMNKXMMMMMMMXNMMMMNXWMMMMMMMMMx  .KMMc   OMMMMMMk  .xMMMWXXWMMMMMMMMWNXXXXKXMMMMMNXKXXWMMMMM
MMMMMMKo,.   .oWXd:. .N0c.  cMMMMMMMMK   'MMN.  .WMMMMWc  ;XMWd,.  .xMMMMKl'      .OMWO:.      :XMMM
MMMMO'  lOx.   . .   ..     0MMMMMMMMX   .KMO   :MMMWd. ,OMWd. :d   :MMO, .ldo.  cWWd. .dOk.  ,MMMMM
MMWl  .KMMM:   xNc   :xK.  lMMMMMMMMMMkd0WMMl   dMWd. 'OMMW;  ,d. .oWMd   0MMl  .XMo   0MMO   kMMMMM
MMo   0MMMX.  lMK   dMMd  .NMMWMMMMMMMMMMMMM,   kl. ,OMMMMd     ;xNMNO   cMMk   cMK   lMMW,  .WMMMMM
MW.  .WMMK.  lWMc  .WMW'  cOo,lMMMMMMMMMMMMK      'OMMMMMM;   oWMWk; .   cl.    0Ml   ONx,   cx;,WMM
MN.   :dc  'kMMK   dMMk     ,dNMMMMMMMMMMMMo    ,0MMMMMMMM;   'c,. ;k:    'c   ,MM:   . .     .c0MMM
MMK:. ..:xXMMMM:.;lNMM0..:xNMMMMMMMMMMMMMMW,..;kMMMMMMMMMMX;.  .;dXWk' .c0Ml   OMM0. .:kW: 'lOWMMMMM
MMMMMWMMMMMMMMWNMMMMMMMWMMMMMMMMMMMMMMMMMMMWMMMMMMMMMMMMMMMMMWWMMWd. 'kWMMX.  ;MMMMMWMMMMMWMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMk.  xWMMMN;  'XMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMX.   .'''.  ,kMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMKd:;,;:coOWMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
EOF



function compile() {
cd ~/daphne
set -x
echo -e "\n\n... Ready to update local GIT code repository with the the latest remote source files ..."

git branch
git pull

echo -e "\n\n... Ready to build the local GIT code repository using a Singularity container ..."
./build-with-singularity-container-ales.sh

echo -e "\n\n... Ready to copy the last build to the container ..."
cd ~/daphne-container/container && rm -rf build/
cp -rp $(ls -dt1 ~/daphne/build_* | head -n 1) build
}
[ "$1" == "-c" ] && compile

time (
cd ~/daphne-container/container-components 
echo -e "\n... Ready to package the build...."
./package.sh

echo -e "\n\n... Ready to deploy the package to Vega and run it (spawn workers and serve demo sequences) ..."
date  +"Time is: "%F+%T
./deploy+run-at-Vega.sh 
) | tee TIME

date  +"Time is: "%F+%T

echo "Comparing correctness of results from these experiments (reprint - 1..5 ONE WORKER, 6..10 ALL WORKERS):"
cat TIME | grep "DenseMatrix(" | awk '{for (i=1; i<=(NF>15?15:NF); i++) printf($i" "); if (NF>15) {printf(" ... ");for (i=(NF-15)>0?NF-25:1; i<=NF; i++) printf($i" "); print""}}'
echo "... only the mixed components:"
cat TIME | grep "DenseMatrix(" | awk '{for (i=1; i<=NF; i++) if ($i-i+2) printf("%s ", $i-i+2); print"";}'

echo -e "\n\n\nThis is the comparison of timings 1-5 (one worker) and 6..10 (all workers):"
cat TIME | grep "seconds for compute" | awk '{print $1}' | tee TIME.g
cat TIME.g | gnuplot -p -e "set terminal dumb size 120, 25; set autoscale; plot '-' with boxes notitle"

echo -e "\n\n* \\ + \\ The Vega DEMO is now complete. Thanks for attending! / + / *"
