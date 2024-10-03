import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { DaphneExecutionComponent } from './main-content/daphne-execution/daphne-execution.component';
import { CommunicationComponent } from './main-content/experiments/communication/communication.component';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { OutputPanelComponent } from './main-content/shared/output-panel/output-panel.component';
import { AggregateStatisticsPanelComponent } from './main-content/shared/aggregate-statistics-panel/aggregate-statistics-panel.component';
import { IoComponent } from './main-content/experiments/io/io.component';
import { HeteroHardwareComponent } from './main-content/experiments/hetero-hardware/hetero-hardware.component';
import { ExperimentsComponent } from './main-content/experiments/experiments.component';

@NgModule({
  declarations: [
    AppComponent,
    DaphneExecutionComponent,
    CommunicationComponent,
    OutputPanelComponent,
    AggregateStatisticsPanelComponent,
    IoComponent,
    HeteroHardwareComponent,
    ExperimentsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    NgbModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
