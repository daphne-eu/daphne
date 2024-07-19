import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class ConfigService {
   // Set your API URL here
  apiUrl: string = 'http://localhost:5000';
  // Set Grafana URL (if any) here
  grafana: {
    isUsed : boolean,
    url: string
  } = {
    isUsed: false, 
    url: ""
  }
}
