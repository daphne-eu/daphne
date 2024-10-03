from flask import current_app
import os
import sys
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import paramiko
from time import sleep
import subprocess

def startWorkers(numberOfWorkers):
    if current_app.config["config"]["use_container"]:
        raise Exception("Running with docker is not supported for Distributed Workers")
    ### Worker list ###
    POSSIBLE_WORKER_IPS=current_app.config["config"]["distributed_workers_list"]
    workers = ','.join([str(i) for i in POSSIBLE_WORKER_IPS[0:numberOfWorkers]])     
    proc = subprocess.Popen(
        ['bash', '{}/deploy/deployDistributed.sh'.format(current_app.config["config"]["paths"]["daphne_dir"]),
        '--run', '--peers', workers
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sleep(1) # wait for workers to spawn 
    return workers

def killAllWorkers(config):
    proc = subprocess.Popen(
        ['bash', '{}/deploy/deployDistributed.sh'.format(config["paths"]["daphne_dir"]),
        '--kill', '--peers', ','.join(config["distributed_workers_list"])
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def getValueFromParams(parameter, daphneParams):
    backendFlag = [i for i in daphneParams if parameter in i]
    if backendFlag:
        return backendFlag[0].split("=")[1]
    else:
        return ""
        
