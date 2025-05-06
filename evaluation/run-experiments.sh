#!/usr/bin/bash
#run file in evaluation dir via ./run-experiments.sh
CSV_DIR="data"
LOG_DIR="./results"

REPS=6
DAPHNE="../bin/daphne"
SKIP_FILENAMES=("frame_1000000r_1000c_MIXED.csv" "frame_1000r_1000000c_MIXED.csv" "frame_100000_10000c_MIXED.csv" "frame_5000000r_100c_MIXED.csv" "frame_5000000r_100c_NUMBER.csv")
# Build a corresponding Daphne script if not yet present.
for csvfile in "$CSV_DIR"/*.csv; do 
    filename=$(basename "$csvfile")
    
    if [[ " ${SKIP_FILENAMES[@]} " =~ " ${filename} " ]]; then
        continue
    fi

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

    # Write headers if these files do not exist.
    # Header: CSVFile,Experiment,Trial,ReadTime,GenerateTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds
    for logfile in "$logNormal" "$logCreate"; do
        if [ ! -f "$logfile" ]; then
            echo "CSVFile,Experiment,Trial,ReadTime,GenerateTime,StartupSeconds,ParsingSeconds,CompilationSeconds,ExecutionSeconds,TotalSeconds" > "$logfile"
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
         # Extract read time from the printed output (using grep with PCRE).
         read_time=$(echo "$output" | grep -oP 'READ_TIME=\K[0-9.]+')
         # If not found, default to 0.
         [ -z "$read_time" ] && read_time="0"
         json_line=$(echo "$output" | grep "{" | head -n1)
         startup=$(echo "$json_line" | grep -oP '"startup_seconds":\s*\K[0-9.]+')
         parsing=$(echo "$json_line" | grep -oP '"parsing_seconds":\s*\K[0-9.]+')
         compilation=$(echo "$json_line" | grep -oP '"compilation_seconds":\s*\K[0-9.]+')
         execution=$(echo "$json_line" | grep -oP '"execution_seconds":\s*\K[0-9.]+')
         total=$(echo "$json_line" | grep -oP '"total_seconds":\s*\K[0-9.]+')
         echo "$filename,normal,$i,$read_time,,${startup},${parsing},${compilation},${execution},${total}" >> "$logNormal"
    done

    ###########################
    # Experiment 2: Read generating meta data
    ###########################
    for i in $(seq 1 $((REPS+1))); do
         metaFile="${csvfile}.meta"
         [ -f "$metaFile" ] && rm -f "$metaFile"
         output=$(stdbuf -oL $DAPHNE --timing "$daphneFile" 2>&1)
         # Discard first run.
         if [ $i -eq 1 ]; then continue; fi
         # Extract metadata generation time and read time.
         meta_time=$(echo "$output" | grep -oP 'GEN_TIME=\K[0-9eE\.-]+')
         [ -z "$meta_time" ] && meta_time="0"
         read_time=$(echo "$output" | grep -oP 'READ_TIME=\K[0-9eE\.-]+')
         [ -z "$read_time" ] && read_time="0"
         json_line=$(echo "$output" | grep "{" | head -n1)
         startup=$(echo "$json_line" | grep -oP '"startup_seconds":\s*\K[0-9.]+')
         parsing=$(echo "$json_line" | grep -oP '"parsing_seconds":\s*\K[0-9.]+')
         compilation=$(echo "$json_line" | grep -oP '"compilation_seconds":\s*\K[0-9.]+')
         execution=$(echo "$json_line" | grep -oP '"execution_seconds":\s*\K[0-9.]+')
         total=$(echo "$json_line" | grep -oP '"total_seconds":\s*\K[0-9.]+')
         
         echo "$filename,create,$i,$read_time,$meta_time,${startup},${parsing},${compilation},${execution},${total}" >> "$logCreate"
         # Remove the meta file before the next iteration.
         metaFile="${csvfile}.meta"
         [ -f "$metaFile" ] && rm -f "$metaFile"
    done

done
echo "Experiments completed."