from flask import current_app
import subprocess
import os
from . import utils
from threading import Thread

class LocalDeployment():
    def __init__(self, parameters) -> None:
        super().__init__()
        self.executionMode = parameters["execution_mode"]
        self.numberOfDistributedNodes = parameters["number_of_distributed_nodes"]
        self.daphneParams = parameters["daphne_params"]
        self.daphneArgs = parameters["daphne_args"]
        self.process = None
    
    
    def isRunning(self):
        if self.process != None:
            return self.process.poll() == None
        return False

    def getOutput(self):
        ret = ""
        with open(current_app.config["config"]["paths"]["stderr_file"], 'r') as f:
            ret = f.read()
        with open(current_app.config["config"]["paths"]["stdout_file"], 'r') as f:
            ret += f.read()
        return ret

    def startExperiment(self):
        my_env = os.environ.copy()
        cmd = []

        backendFlag = utils.getValueFromParams("dist_backend", self.daphneParams)
        if self.executionMode == "distributed":
            # check for gRPC or mpi
            if ("gRPC" in backendFlag):
                # Start workers
                workers = utils.startWorkers(self.numberOfDistributedNodes)
                my_env["DISTRIBUTED_WORKERS"] = workers
            elif ("MPI" in backendFlag):
                # Modify command
                cmd += ["mpirun"] + ["-np"] + [str(self.numberOfDistributedNodes + 1)] # for MPI + 1 since 1 is coordinator

        # Check for container
        if current_app.config["config"]["use_container"]:
            cmd += current_app.config["config"]["container_cmd"].split(" ")            
        else: # natively
            cmd += ['./bin/daphne']        

        cmd += self.daphneParams
        cmd += self.daphneArgs.split(" ")

        # remove empty elements
        cmd = [c for c in cmd if c != '']
        
        # Start execution
        with open(current_app.config["config"]["paths"]["stdout_file"], 'w') as out_f, open(current_app.config["config"]["paths"]["stderr_file"], 'w') as err_f:
            out_f.write(' '.join(cmd))
            out_f.write("\n")
            self.process = subprocess.Popen(cmd, cwd=current_app.config["config"]["paths"]["daphne_dir"], stdout=out_f, stderr=err_f, env=my_env)
            if self.executionMode == "distributed" and "gRPC" in backendFlag:
                self.terminateWorkers(current_app.config["config"])
            return 
        
    def terminateWorkers(self, config):
        if self.process is not None:
            self.process.wait()
        utils.killAllWorkers(config)
    
    def kill(self):
        Thread(target=self.terminateWorkers, args=[current_app.config["config"]]).start()
        self.process.kill()