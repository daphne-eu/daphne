from flask import current_app
import paramiko
from threading import Thread
from . import utils

class SSHClient(paramiko.SSHClient):

    def handler(self, title, instructions, prompt_list):
        for item in prompt_list:
            if "Verification" in item[0]:
                return [str(self.totp)]
        return []

    def auth_interactive(self, username, handler):
        if not self.totp:
            raise ValueError('Need a verification code for 2fa.')
        self._transport.auth_interactive(username, handler)

    def _auth(self, username, password, pkey, *args):
        self.password = password
        two_factor = False
        allowed_types = set()
        two_factor_types = {'keyboard-interactive', 'password', 'publickey'}

        agent = paramiko.Agent()
        try:
            agent_keys = agent.get_keys()
        except:
            pass

        for key in agent_keys:
            try:
                self._transport.auth_publickey(username, key)
                return
            except paramiko.SSHException as e:
                print(e)

        if pkey is not None:
            try:
                allowed_types = set(
                    self._transport.auth_publickey(username, pkey)
                )
                two_factor = allowed_types & two_factor_types
                if not two_factor:
                    return
            except paramiko.SSHException as e:
                print(e)

        return self.auth_interactive(username, self.handler)     

class VegaDeployment():
    def __init__(self, parameters) -> None:
        super().__init__()
        self.executionMode = parameters["execution_mode"]
        self.numberOfDistributedNodes = parameters["number_of_distributed_nodes"]
        self.daphneParams = parameters["daphne_params"]
        self.daphneArgs = parameters["daphne_args"]
        self.token = parameters["vega_token"]
        
        self.client = self.createClient()
        self.output = []
    
    def isRunning(self):
        _, ret, _ = self.client.exec_command('squeue -n DaphneCoordinator')
        if (len(ret.readlines()) > 1):
            return True
        return False

    def getOutput(self):
        if self.client != None:
            _, self.stdout, _ = self.client.exec_command("cat {dir}/{out}; cat {dir}/{err}".format(
                dir=current_app.config["config"]["vega_config"]["daphne_dir"],
                out=current_app.config["config"]["vega_config"]["stdout_file"],
                err=current_app.config["config"]["vega_config"]["stderr_file"]
            ))
            self.output = ''.join(self.stdout.readlines())
            return self.output
        
    def createClient(self):
        current_app.config["config"]["vega_config"]
        host = current_app.config["config"]["vega_config"]["host"]
        port = current_app.config["config"]["vega_config"]["port"]
        username = current_app.config["config"]["vega_config"]["username"]

        # Load the RSA private key
        private_key = paramiko.RSAKey(filename=current_app.config["config"]["vega_config"]["private_key_path"])


        client = SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy)
        
        client.totp = self.token
        client.connect(hostname=host, port=port, username=username, pkey=private_key)
        return client



    def startExperiment(self):
        self.output = []
        programParam = ['bin/daphne']

        distributedBackendFlag = utils.getValueFromParams("dist_backend", self.daphneParams)

        programParam += self.daphneParams
        programParam += self.daphneArgs.split(" ")
        
        coresPerTask = 1
        try:
            coresPerTask = int(utils.getValueFromParams("num-threads", self.daphneParams))
        except ValueError:
            coresPerTask = 1
        
        if self.executionMode == "distributed":
            # check for gRPC or mpi
            if ("gRPC" in distributedBackendFlag):
                raise Exception("Distributed backend gRPC on VEGA not implemented") 
                # pass # spawn workers on vega
            if ("MPI" in distributedBackendFlag):
                mpi_batch = ''.join(open("scripts/batch_mpi.txt", 'r').readlines()).format(
                    daphne_dir=current_app.config["config"]["vega_config"]["daphne_dir"],
                    distributed_nodes=self.numberOfDistributedNodes + 1,
                    coresPerTask=coresPerTask, 
                    daphneParams=' '.join(self.daphneParams),
                    daphneArgs=' '.join(self.daphneArgs.split(" ")),
                    out_f=current_app.config["config"]["vega_config"]["daphne_dir"] + "/" + current_app.config["config"]["vega_config"]["stdout_file"],
                    err_f=current_app.config["config"]["vega_config"]["daphne_dir"] + "/" + current_app.config["config"]["vega_config"]["stderr_file"]
                )
        # Clear last logs
        cmd = "rm {dir}/{out} {dir}/{err}; ".format(
            dir=current_app.config["config"]["vega_config"]["daphne_dir"],
            out=current_app.config["config"]["vega_config"]["stdout_file"],
            err=current_app.config["config"]["vega_config"]["stderr_file"]
        )
        if self.executionMode == "distributed" and "MPI" in distributedBackendFlag:
            cmd += "sbatch <<EOT\n" + mpi_batch + "\nEOT\n; "            
        else:
            cmd += "cd {dir}; srun --job-name=DaphneCoordinator --time=01:00:00 --cpus-per-task={coresPerTask} --ntasks=1 singularity exec --env LD_LIBRARY_PATH=\`pwd\`/lib:\`pwd\`/thirdparty/installed/lib:$LD_LIBRARY_PATH daphne.sif {arg} > {out} 2> {err}; "\
            .format(
                dir=current_app.config["config"]["vega_config"]["daphne_dir"],
                coresPerTask=coresPerTask, 
                arg=' '.join(programParam),
                out=current_app.config["config"]["vega_config"]["stdout_file"],
                err=current_app.config["config"]["vega_config"]["stderr_file"]
            )
        # Escape ""
        cmd = cmd.replace('"', "\\\"")
        # Echo command to output file.
        cmd += "echo {} > {}".format(cmd, current_app.config["config"]["vega_config"]["stdout_file"])
        self.client.exec_command(cmd)

    def terminateWorkers(self):
        # Terminate workers vega 
        self.client.exec_command("scancel -n DaphneDistributedWorkers")
    
    def kill(self):
        Thread(target=self.terminateWorkers).start()
        self.client.exec_command("scancel -n DaphneCoordinator")
        return
