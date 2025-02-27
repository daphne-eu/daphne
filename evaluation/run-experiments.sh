#!/usr/bin/bash
#run file in evaluation dir via ./run-experiments.sh
CSV_DIR="data"
LOG_DIR="./results"

REPS=6
DAPHNE="../bin/daphne"

# Build a corresponding Daphne script if not yet present.
for csvfile in "$CSV_DIR"/*.csv; do 
    filename=$(basename "$csvfile")

    # Determine if the CSV is a matrix or a frame based on its name.
    if [[ $filename == matrix_* ]]; then
       fileType="matrix"
    else
       fileType="frame"
    fi
    # Build a corresponding Daphne script if not yet present.
    daphneFile="${CSV_DIR}/${filename%.csv}.daphne"
    if [ ! -f "$daphneFile" ]; then
        if [ "$fileType" == "matrix" ]; then
            echo "readMatrix(\"${csvfile}\");" > "$daphneFile"
        else
            echo "readFrame(\"${csvfile}\");" > "$daphneFile"
        fi
    fi

    # Choose proper log file names.
    logNormal="${LOG_DIR}/evaluation_results_${filename}_normal.csv"
    logCreate="${LOG_DIR}/evaluation_results_${filename}_create.csv"
    logOpt="${LOG_DIR}/evaluation_results_${filename}_opt.csv"

    # Write headers if these files do not exist.
    # Header: CSVFile,Experiment,Trial,ReadTime,WriteTime,PosmapReadTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds
    for logfile in "$logNormal" "$logCreate" "$logOpt"; do
        if [ ! -f "$logfile" ]; then
            echo "CSVFile,Experiment,Trial,ReadTime,WriteTime,PosmapReadTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds" > "$logfile"
        fi
    done

    echo "Running experiments for $filename ..."

    ###########################
    # Experiment 1: Normal Read
    ###########################
    for i in $(seq 1 $((REPS+1))); do
         output=$(stdbuf -oL $DAPHNE --timing "$daphneFile" 2>&1)
         # Discard the first run (warm-up).
         if [ $i -eq 1 ]; then continue; fi
         # Use a sed pattern that captures floating‐point numbers.
         read_time=$(echo "$output" | head -n1 | sed 's/[^0-9.]*\([0-9.]*\).*/\1/')
         json_line=$(echo "$output" | tail -n1)
         startup=$(echo "$json_line" | grep -oP '"startup_seconds":\s*\K[0-9.]+')
         parsing=$(echo "$json_line" | grep -oP '"parsing_seconds":\s*\K[0-9.]+')
         compilation=$(echo "$json_line" | grep -oP '"compilation_seconds":\s*\K[0-9.]+')
         execution=$(echo "$json_line" | grep -oP '"execution_seconds":\s*\K[0-9.]+')
         total=$(echo "$json_line" | grep -oP '"total_seconds":\s*\K[0-9.]+')
         echo "$filename,normal,$i,$read_time,,,${startup},${parsing},${compilation},${execution},${total}" >> "$logNormal"
    done

    ###########################
    # Experiment 2: First Read (Create Posmap)
    ###########################
    for i in $(seq 1 $((REPS+1))); do
          posmapFile="${csvfile}.posmap"
         [ -f "$posmapFile" ] && rm -f "$posmapFile"
         # Always use --second-read-opt for posmap creation.
         output=$(stdbuf -oL $DAPHNE --timing --use-positional-map "$daphneFile" 2>&1)
         # Discard first run.
         if [ $i -eq 1 ]; then continue; fi
         # Extract overall read time from READ_TYPE=first and write posmap time from OPERATION=write_posmap.
         read_line=$(echo "$output" | grep "READ_TYPE=first," | head -n1)
         read_time=$(echo "$read_line" | sed 's/.*READ_TIME=\([0-9eE\.-]*\).*/\1/')
         [ -z "$read_time" ] && read_time="0"
         write_line=$(echo "$output" | grep "OPERATION=write_posmap," | head -n1)
         write_time=$(echo "$write_line" | sed 's/.*WRITE_TIME=\([0-9eE\.-]*\).*/\1/')
         [ -z "$write_time" ] && write_time="0"
         json_line=$(echo "$output" | grep "{" | head -n1)
         startup=$(echo "$json_line" | grep -oP '"startup_seconds":\s*\K[0-9.]+')
         parsing=$(echo "$json_line" | grep -oP '"parsing_seconds":\s*\K[0-9.]+')
         compilation=$(echo "$json_line" | grep -oP '"compilation_seconds":\s*\K[0-9.]+')
         execution=$(echo "$json_line" | grep -oP '"execution_seconds":\s*\K[0-9.]+')
         total=$(echo "$json_line" | grep -oP '"total_seconds":\s*\K[0-9.]+')
         # For first read, we report overall ReadTime and WriteTime; PosmapReadTime remains empty.
         echo "$filename,create,$i,$read_time,$write_time,,${startup},${parsing},${compilation},${execution},${total}" >> "$logCreate"
         # Remove the posmap file before the next iteration.
         posmapFile="${csvfile}.posmap"
         [ -f "$posmapFile" ] && rm -f "$posmapFile"
    done

    ###########################
    # Experiment 3: Second Read (Optimized Read using posmap)
    ###########################
    # First, create the posmap.
    posmapFile="${csvfile}.posmap"
    $DAPHNE --timing --second-read-opt "$daphneFile" > /dev/null
    # Reuse the posmap for each trial.
    for i in $(seq 1 $((REPS+1))); do
         output=$(stdbuf -oL $DAPHNE --timing --use-positional-map "$daphneFile" 2>&1)
         if [ $i -eq 1 ]; then continue; fi
         # Extract posmap read time (this line comes first).
         posmap_line=$(echo "$output" | grep "OPERATION=read_posmap," | head -n1)
         posmap_read_time=$(echo "$posmap_line" | sed 's/.*READ_TIME=\([0-9eE\.-]*\).*/\1/')
         [ -z "$posmap_read_time" ] && posmap_read_time="0"
         # Extract overall read time directly with grep -oP.
         read_line=$(echo "$output" | grep "READ_TYPE=second," | head -n1)
         read_time=$(echo "$read_line" | sed 's/.*READ_TIME=\([0-9eE\.-]*\).*/\1/')
         [ -z "$read_time" ] && read_time="0"
         json_line=$(echo "$output" | grep "{" | head -n1)
         startup=$(echo "$json_line" | grep -oP '"startup_seconds":\s*\K[0-9.]+')
         parsing=$(echo "$json_line" | grep -oP '"parsing_seconds":\s*\K[0-9.]+')
         compilation=$(echo "$json_line" | grep -oP '"compilation_seconds":\s*\K[0-9.]+')
         execution=$(echo "$json_line" | grep -oP '"execution_seconds":\s*\K[0-9.]+')
         total=$(echo "$json_line" | grep -oP '"total_seconds":\s*\K[0-9.]+')
         echo "$filename,opt,$i,$read_time,,${posmap_read_time},${startup},${parsing},${compilation},${execution},${total}" >> "$logOpt"
    done

done
echo "Experiments completed."