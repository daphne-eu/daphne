from flask import Blueprint, request, jsonify, current_app
import subprocess
import os
from psutil import Process
from tools.LocalDeployment import LocalDeployment
from tools.VegaDeployment import VegaDeployment

api_bp = Blueprint("api", __name__)

script_process = None

deployment = None


@api_bp.route("/run_daphne", methods=["POST"])
def run_daphne():
    received_data = request.json
    global deployment

    try:
        if received_data["cluster"] == "local_machine":
            deployment = LocalDeployment(received_data)
        elif received_data["cluster"] == "vega":
            deployment = VegaDeployment(received_data)
        else:
            return jsonify(success=False, message='Unknown cluster selected.')    

        deployment.startExperiment()
        return jsonify(success=True, message='Script started executing.')

    except Exception as e:
        return jsonify(success=False, message=str(e))



def getStatus():
    global deployment
    if deployment is not None:
        return deployment.isRunning()
    return False


@api_bp.route('/job_status', methods=['GET'])
def job_status():
    global script_process
    global deployment
    if deployment is not None:
        return jsonify(success=True, running=deployment.isRunning())
    return jsonify(success=True, running=False)


@api_bp.route('/get_output', methods=['GET'])
def get_output():
    global deployment
    ret = {}
    ret["output"] = ""
    if deployment is not None:
        ret["output"] = deployment.getOutput()
    return jsonify(success=True, message={"running": getStatus(), "output": ret})

@api_bp.route('/kill_daphne', methods=['POST'])
def kill_script():
    global deployment
    if deployment is not None and deployment.isRunning():
        deployment.kill()
        return jsonify(success=True, message='Deployment killed.')
    return jsonify(success=True, message='No deployment found.')  


@api_bp.route("get_setup_settings", methods=["GET"])
def get_algorithms():
    res = {}
    res["algorithm_list"] = current_app.config["config"]["algorithms"]["algorithm_list"]
    res["daphne_options"] = current_app.config["config"]["daphne_options"]
    res["max_distributed_workers"] = len(current_app.config["config"]["distributed_workers_list"])
    return jsonify(success=True, message=res)