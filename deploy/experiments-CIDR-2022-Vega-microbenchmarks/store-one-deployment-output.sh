#!/bin/bash

[ "$1" == "-10" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_10/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-04-NUMCORES_10_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh 
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
)

[ "$1" == "-500" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_500/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_500_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
)

[ "$1" == "-1000" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_1000/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_1000_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
)
)

[ "$1" == "-100" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_100/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_100_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
)

[ "$1" == "-500" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_500/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_500_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
)

[ "$1" == "-1000" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_1000/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_1000_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
cp -p TIME.g ${logfile}.timing
cp -p TIME ${logfile}.out
cat ${logfile}.timing | gnuplot -p -e "set terminal pngcairo enhanced; set output '"${logfile}.timing.png"'; set autoscale; set xlabel 'run number: 1-5 (one node) and 6-10 (1000 DistributedWorker nodes)'; set xtics 1; set xrange [0.5:10.5]; set ylabel 'time [s]'; set key below; set style fill solid 0.8; plot '-' u (\$0+1):1 with boxes lw 2 lc rgb '#7889fb' title 'The components algorithm execution statistics.'"
)
