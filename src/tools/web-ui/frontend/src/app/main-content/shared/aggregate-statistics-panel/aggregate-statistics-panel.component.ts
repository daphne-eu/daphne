import { Component, Input, SimpleChanges } from '@angular/core';
import { ConfigService } from 'src/app/core/services/config.service';


@Component({
  selector: 'app-aggregate-statistics-panel',
  templateUrl: './aggregate-statistics-panel.component.html',
  styleUrls: ['./aggregate-statistics-panel.component.scss']
})
export class AggregateStatisticsPanelComponent {
  @Input() text: string = "";
  statistics ;

  constructor(public configService: ConfigService){
    if(this.text)
      this.statistics = this.parseStatistics(this.text);
  }
  ngOnChanges(changes: SimpleChanges) {
    // Check if the 'inputValue' property has changed
    if (changes['text']) {
      if(changes['text'].currentValue)
        this.statistics = this.parseStatistics(changes['text'].currentValue);      
      else 
        this.statistics = new Map();
    }
  }
  parseStatistics(inputString: string) {
    const lines = inputString.trim().split('\n');
    let result = new Map(lines.filter(line => {
        const parts = line.split('}').map(item => item.trim());             
        const phaseMatch = parts[0].match(/phase="([^"]+)"/);
        if (phaseMatch)
          return true;      
        return false;
    }).map(line => {
        const parts = line.split('}').map(item => item.trim());      
        const label = parts[0].split('{')[0];
        const phaseMatch = parts[0].match(/phase="([^"]+)"/);
        const phase = phaseMatch ? phaseMatch[1] : "unknown";        
        const numberStr = parts[1];
        return [phase, this.formatNumber(phase, numberStr)];
    }));    
    const execTime = result.get("execution_seconds") || "0";
    const readTime = result.get("read_seconds") || "0";
    let realExecTime = (parseFloat(execTime) - parseFloat(readTime)).toFixed(3).toString();
    result.set("execution_seconds", realExecTime + " sec");
    return result;
  }

  formatNumber(label: string, number: string): string {
    let num = parseFloat(number);
    
    if (label.includes('Bytes')) {      
      let bytes = parseInt(num.toString(), 10);
      if (bytes < 1024) {
          return `${bytes} bytes`;
      } else if (bytes < 1024 * 1024) {
          return `${(bytes / 1024).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")} KB`;
      } else if (bytes < 1024 * 1024 * 1024) {
          return `${(bytes / (1024 * 1024)).toFixed(3)} MB`;
      } else if (bytes < 1024 * 1024 * 1024 * 1024) {
          return `${(bytes / (1024 * 1024 * 1024)).toFixed(3)} GB`;
      } else {
          return `${(bytes / (1024 * 1024 * 1024 * 1024)).toFixed(3)} TB`;
      }
    } else {
      return `${num.toFixed(3)} sec`
    }
  }
}
