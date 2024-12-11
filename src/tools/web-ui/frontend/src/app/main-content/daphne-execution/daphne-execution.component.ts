import { Component } from '@angular/core';
import { ExecutionMode } from '../experiments/hetero-hardware/hetero-hardware.component';
import { Cluster } from '../experiments/hetero-hardware/hetero-hardware.component';
import { DistributedBackend } from 'src/app/core/models/distributed-backend';
import { ConfigService } from 'src/app/core/services/config.service';
import { DataService } from 'src/app/core/services/data.service';

@Component({
  selector: 'app-daphne-execution',
  templateUrl: './daphne-execution.component.html',
  styleUrls: ['./daphne-execution.component.scss']
})

export class DaphneExecutionComponent {
  DistributedBackend = DistributedBackend;
  apiUrl: string;
  Algorithms = undefined;
  ExecutionMode = ExecutionMode;
  Cluster = Cluster;

  // Var for selected output
  selected_output_panel = "output-panel";

  // Scheduling
  partitioning_options: any[] = [];
  queue_layout_options: any[] = [];
  victim_selection_options: any[] = [];
   
  // Max limits
  maxCores = 2;
  maxNumberNodes = 1;
  maxLocalCores = 2;
  maxLocalNumberNodes = 1;

  // Form
  cluster: Cluster = Cluster.local_machine;
  executionMode: ExecutionMode = ExecutionMode.single_node;
  numberOfNodes: number = 2;
  coresPerNode: number = 2;
  vectorization: boolean = false;
  partitioning: string = "";
  queue_layout: string = "";
  victim_selection: string = "";
  select_matrix_repr: boolean = false;
  dist_backend: DistributedBackend = DistributedBackend.mpi;
  use_cuda: boolean = false;
  selectedAlgorithm: any;
  inputSize: string = "";
  vega_token = "";

  result_output : string = "";
  agg_statistics = "";

  running_daphne = false;

  // Alerts
  show_success_alert = false;
  show_canceled_alert = false;
  // Errors
  show_error_message = false;
  error_message = "";

  constructor(private dataService: DataService, private configService: ConfigService) {
    this.apiUrl = this.configService.apiUrl;
    this.killDaphne();

    // Fetch possible scripts/algorithms
    this.dataService.getSetupSettings().subscribe({
      next: (res: any) => {
        this.Algorithms = res.algorithm_list
        this.selectedAlgorithm = res.algorithm_list[0]
        this.inputSize = res.algorithm_list[0].arguments[0]?.arguments ?? "";
        this.maxLocalCores = parseInt(res.daphne_options.max_cpus)
        this.maxLocalNumberNodes = res.max_distributed_workers

        this.maxCores = this.maxLocalCores;
        this.maxNumberNodes = this.maxLocalNumberNodes;        

        this.partitioning_options = res.daphne_options.partitioning
        this.queue_layout_options = res.daphne_options.queue_layout
        this.victim_selection_options = res.daphne_options.victim_selection
      },
      error: (err) => this.displayError(err)
    });
  }

  onSubmit() {
    // Create a payload object with the form data
    const payload : {
      cluster: Cluster,
      execution_mode: ExecutionMode,
      number_of_distributed_nodes: number,
      daphne_params: string[],
      daphne_args: string,
      vega_token: number
    }
    = {
      cluster: this.cluster,
      execution_mode: this.executionMode,
      number_of_distributed_nodes: this.numberOfNodes,
      daphne_params: [],
      daphne_args: "",
      vega_token: parseInt(this.vega_token)
    };
    if (this.vectorization){
      payload.daphne_params.push("--vec")
      payload.daphne_params.push("--num-threads=" + this.coresPerNode.toString())
    }
    if (this.use_cuda)
      payload.daphne_params.push("--cuda")
    if (this.executionMode == ExecutionMode.distributed){
      payload.daphne_params.push("--distributed")
      payload.daphne_params.push("--dist_backend=" + this.dist_backend)
      // Todo add mpi-grpc selection option
    }
    if (this.select_matrix_repr)
      payload.daphne_params.push("--select-matrix-repr")
    payload.daphne_args += this.selectedAlgorithm.filepath + " " + this.inputSize
    
    // Scheduling knobs
    if (this.partitioning)
        payload.daphne_params.push(this.partitioning)
      if (this.queue_layout)
        payload.daphne_params.push(this.queue_layout)
      if (this.victim_selection)
        payload.daphne_params.push(this.victim_selection)
    // Clear output
    this.result_output = "";
    // Set flag
    this.running_daphne = true
    // Make a POST request to the API with the payload
    this.dataService.runDaphne(payload).subscribe({
        next: (res: any) => {
          this.getResults();
        },
        error: (err) => {
          this.killDaphne();
          this.running_daphne = false;
          this.displayError(err)
        }
      });
  }

  getResults() {
    this.dataService.getOutput().subscribe({
      next: (res: any) => {
        if (!this.running_daphne){
          // Return recursively
          return;
        } 

        if(!res.running) {
          this.running_daphne = false;
          this.show_success_alert = true;
          setTimeout(() => {
            this.show_success_alert = false;
          }, 3000);
        }
        this.result_output = res.output.output;
        this.agg_statistics = res.output.aggregate_statistics;

        // If we reach this point.. we still need to fetch data, call recursively, after 1s
        setTimeout(() => this.getResults(), 1000);
      }, 
      error: (err) => this.displayError(err)
    })
    // }, 500);
  }

  killDaphne(){
    this.dataService.killDaphne().subscribe({
      next: (res: any) => {
        this.running_daphne = false;
        this.show_canceled_alert = true;
        setTimeout(() => {
          this.show_canceled_alert = false;
        }, 3000);        
      }, 
      error: (err) => this.displayError(err)
    })
  }
  updateLimits() {
    if (this.cluster === Cluster.local_machine){
      this.maxCores = this.maxLocalCores;
      this.maxNumberNodes = this.maxLocalNumberNodes;
    }
    else { // vega 
      this.maxCores = 64;
      this.maxNumberNodes = 32;
    }
  }
  onChange(newAlgorithm: any) {
    this.selectedAlgorithm = newAlgorithm;
    this.inputSize = newAlgorithm.arguments[0]?.arguments ?? "";
  }

  private displayError(err: Error){
    this.error_message = err.message
    this.show_error_message = true;
    setTimeout(() => {
      this.show_error_message = false;
    }, 3000);
  }
}