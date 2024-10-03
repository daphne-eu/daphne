import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { ConfigService } from './config.service';
import { catchError, map, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  apiUrl: string;

  constructor(private http: HttpClient, private config: ConfigService) {
    this.apiUrl = this.config.apiUrl;
  }

  runDaphne(payload: any){
    return this.http.post(`${this.apiUrl}/api/run_daphne`, payload).pipe(      
      catchError(this.handleError),
      map(this.isSuccessful)
    )
  }
  
  killDaphne(){
    return this.http.post(`${this.apiUrl}/api/kill_daphne`, null).pipe(
      catchError(this.handleError),
      map(this.isSuccessful)
    )
  }

  getOutput(){
    return this.http.get(`${this.apiUrl}/api/get_output`).pipe(
      catchError(this.handleError),
      map(this.isSuccessful)
    )
  }

  getSetupSettings() {
    return this.http.get(`${this.apiUrl}/api/get_setup_settings`).pipe(
      catchError(this.handleError),
      map(this.isSuccessful)
    )
  }
  
  private isSuccessful(res: any){
    if (!res.success)
      throw new Error(res.message)
    return res.message
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'An error occurred';
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Client Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Server Error.\n Error Code: ${error.status}.\nMessage: ${error.message}`;
    }
    return of({success: false, message: errorMessage})
  }

}
