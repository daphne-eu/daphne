import { Component } from '@angular/core';
import { ConfigService } from 'src/app/core/services/config.service';
import { DataService } from 'src/app/core/services/data.service';

export enum ExecutionMode{
  single_node = "single_node", 
  distributed = "distributed"
}

export enum Cluster {
  local_machine = "local_machine",
  vega = "vega"
}
@Component({
  selector: 'app-hetero-hardware',
  templateUrl: './hetero-hardware.component.html',
  styleUrls: ['./hetero-hardware.component.scss']
})
export class HeteroHardwareComponent {
  apiUrl: string;
  ExecutionMode = ExecutionMode;
  Cluster = Cluster;

  // Var for selected output
  selected_output_panel = "output-panel";

  // Max limits
  maxCores = 20;
  maxNumberNodes = 8;

  // Form
  cluster: Cluster = Cluster.vega;
  executionMode: ExecutionMode = ExecutionMode.single_node;
  numberOfNodes: number = 4;
  coresPerNode: number = 4;
  cuda_enabled: boolean = false;
  inputSize: string = "small";
  vega_token = "";

  result_output : string = "";
  agg_statistics = "";

  running_daphne = false;

  // Alerts
  show_success_alert = false;
  show_canceled_alert = false;
  constructor(private dataService: DataService, private configService: ConfigService) {
    this.apiUrl = configService.apiUrl;
    this.killDaphne();
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
      daphne_args: 'scripts/algorithms/pagerank.daph maxi=1000 e=0 verbose=false df=0.0002 G=\\\\"\\"\"datasets/pagerank/amazon_x1.mtx\\\\"\\"\"',
      vega_token: parseInt(this.vega_token)
    };
    payload.daphne_params = "--select-matrix-repr --vec --timing".split(" ")
    if (this.cuda_enabled)
      payload.daphne_params.push("--cuda")
    // Clear output
    this.result_output = "";
    // Set flag
    this.running_daphne = true
    // Make a POST request to the API with the payload
    this.dataService.runDaphne(payload)
      .subscribe({
        next: (res: any) => {
          this.getResults();
        },
        error: (err) => {
          this.killDaphne();
          this.running_daphne = false;
          throw new Error(JSON.stringify(err));
        }
      });
  }

  getResults() {
    var resultInterval = setInterval(() => {
      this.dataService.getOutput().subscribe({
      next: (res: any) => {
        if (!this.running_daphne){
          clearInterval(resultInterval);
          return;
        } 

        if(!res.running) {
          this.running_daphne = false;
          clearInterval(resultInterval);
          this.show_success_alert = true;
          setTimeout(() => {
            this.show_success_alert = false;
          }, 3000);
        }
        this.result_output = res.output.output;
        this.agg_statistics = res.output.aggregate_statistics;
      }
    })
    }, 500);
  }

  killDaphne(){
    this.dataService.killDaphne().subscribe({
      next: (res: any) => {
        this.running_daphne = false;
        if (res === "Deployment killed."){
          this.show_canceled_alert = true;
          setTimeout(() => {
            this.show_canceled_alert = false;
          }, 3000);
          }
        }, 
      error: (err) => {
        console.log(err)
      }      
    })
  }  
}
