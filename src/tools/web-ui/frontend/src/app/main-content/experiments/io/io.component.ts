import { Component } from '@angular/core';
import { ConfigService } from 'src/app/core/services/config.service';
import { DataService } from 'src/app/core/services/data.service';

export enum ExecutionMode{
  single_node = "single_node", 
  distributed = "distributed"
}
@Component({
  selector: 'app-io',
  templateUrl: './io.component.html',
  styleUrls: ['./io.component.scss']
})
export class IoComponent {
  apiUrl: string;
  ExecutionMode = ExecutionMode;

  // Var for selected output
  selected_output_panel = "output-panel";

  numberOfNodes = 8;
  executionMode = ExecutionMode.single_node;
  inputSize: string = "small";

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
    const payload = {
      executionMode: this.executionMode,
      inputSize: this.inputSize
    };
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
        console.log(err);
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
        this.result_output = res.output.output
        this.agg_statistics = res.output.aggregate_statistics;
      }
    })
    }, 500);
  }

  killDaphne(){
    this.dataService.killDaphne().subscribe({
      next: (res: any) => {
        this.running_daphne = false;
        if (res.success) {
        if (res === "Deployment killed."){
            this.show_canceled_alert = true;
            setTimeout(() => {
              this.show_canceled_alert = false;
            }, 3000);
          }
        }
      }, 
      error: (err) => {
        console.log(err)
      }
    })
  }
}
