import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-output-panel',
  templateUrl: './output-panel.component.html',
  styleUrls: ['./output-panel.component.scss']
})
export class OutputPanelComponent {
  @Input() text: string = "";
}
