# Daphne UI

This tool is used as a web interface for running DAPHNE either locally or at VEGA HPC. 

**Note that DAPHNE must be already build and ready to run before using the Web UI.**

## Requirements

In order to run the UI you need to install Node.js and npm package manager for the frontend and Python along with flask for the backend API. 

| tool/lib                             | version known to work (*)    | comment                                                                                                                                 |
|--------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Node.js       | 18.16.0   | |
| npm           | 9.5.1
| Python        | 3.8.10    ||
| Flask         | 2.3.2     || 

## Structure

The tool consists of a Python Flask API server (the [backend](./backend/)) which spawns and controls DAPHNE jobs and an Angular application (the [frontend](./frontend/)) which the user interacts with through a browser.

## Web-UI

The frontend is built using Angular Framework. You can read more about how to run the angular app [here](./frontend/README.md). During development the frontend can be served by a seperate node server (using `ng serve`).

## API

The Flask API server controls DAPHNE jobs and responds with outputs/errors to the frontend. You can read more on how to run the backend API [here](./backend/README.md).

## Configuration

[./backend/config.json](./backend/config.json) needs to be configured in order to use the UI. 
Read more [here](./backend/README.md#Configuration)
