#!/bin/bash

[ "$1" == "-10" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_10/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-04-NUMCORES_10_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh 
)

[ "$1" == "-100" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_100/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_100_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
)

[ "$1" == "-500" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_1000/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_500_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
)

[ "$1" == "-1000" ] && (
logfile=daphnec-outputs/components-42-time/n_6000/e_30/NUMCORES_1000/output-$(date +%F_%T).log
echo Logging to $logfile
screen -L -Logfile $logfile ./Demo_on_Vega-2021-12-06-NUMCORES_1000_DAPHNE-components-42-time-N_6000_e_30-Ales_Zamuda.sh
)
