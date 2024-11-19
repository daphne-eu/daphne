import { Component, NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { compileClassMetadata } from '@angular/compiler';
import { DaphneExecutionComponent } from './main-content/daphne-execution/daphne-execution.component';
import { CommunicationComponent } from './main-content/experiments/communication/communication.component';
import { IoComponent } from './main-content/experiments/io/io.component';
import { HeteroHardwareComponent } from './main-content/experiments/hetero-hardware/hetero-hardware.component';
import { ExperimentsComponent } from './main-content/experiments/experiments.component';

const routes: Routes = [
  {path: '', redirectTo: 'daphne-execution', pathMatch: 'full'},
  {path: 'daphne-execution', component: DaphneExecutionComponent},
  {path: 'experiments', component: ExperimentsComponent, children: [
    {path: 'communication', component: CommunicationComponent},
    {path: 'io', component: IoComponent},
    {path: 'hetero-hardware', component: HeteroHardwareComponent}
  ]},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
