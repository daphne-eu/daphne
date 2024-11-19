# Flask API

The Flask API used for running DAPHNE jobs with the Web UI tool.

## Getting started

You can easily install the required dependencies with:
```
pip3 install -r requirements.txt
```

## Configuration

[./backend/config.json](./backend/config.json) needs to be configured in order to use the API. It is necessary to specify:

| JSON key                           | Description               | |
|--------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| paths.daphne_dir       | Path to the DAPHNE directory. |
| use_container          | True/False wether you want to run with containers or natively.
| container_cmd          | This should be the command you use to run DAPHNE with containers. If you use scripts from [containers](/containers/), you can use them here too (e.g. if you run with: "`sudo containers/quickstart.sh example.daph`", change `container_cmd` to `sudo containers/quickstart.sh`).
| algorithms             | Here we specify the available options that will be displayed on the UI. `name` is the name of the algorithm displayed, `filepath` is the path to the script that will be executed (absolute or relative to `paths.daphne_dir`) and `arguments` are the arguments needed by the DAPHNE script in order to be executed.
| vega_config           | SSH configurations for accessing VEGA HPC.
| distributed_workers_list | List of Distributed gRPC workers to deploy when using gRPC backend. It requires [`deploy/deployDistributed.sh`](/deploy/deployDistributed.sh) to be configured.

## Running the API

After installing flask, the API will run with:
```
python3 app.py
```
or using flask command:
```
flask run
```
Note that for `flask` to work, it has to be added to `PATH` variable otherwise the full path can be specified.

# Structure

A brief summary of the API structure:

- `app.py`: The API entrypoint.
- `api.py`: Contains all the API endpoints.
- `tools/LocalDeployment.py`: A class designed to spawn and report on DAPHNE jobs.
- `tools/VegaDeployment.py`: A class designed to spawn and report on DAPHNE jobs on VEGA HPC (it uses paramiko, an ssh library to connect to VEGA and run jobs using slurm commands).
- `scripts/`: A few helper scripts (e.g. for running MPI jobs on VEGA)
