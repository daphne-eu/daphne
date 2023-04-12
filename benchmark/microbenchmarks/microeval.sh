#!/bin/bash

#******************************************************************************
# Utility functions
#******************************************************************************

function print_help () {
    # TODO Use the real defaults (as defined before the argument parsing) here.
    echo "Usage: $0 [-h] [--lm] [--kmeans] [--check] [--run] [-r N]"
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help         Print this help message and exit."
    echo "  --lm               Run linear regression model training, default: "
    echo "                     off."
    echo "  --kmeans           Run k-means clustering, default: off."
    echo "  --check            Compare results of all approaches on the same "
    echo "                     small input data to check for correctnes, "
    echo "                     default: off."
    echo "  --run              Run the actual micro benchmark, default: off."
    echo "  -r, --repetitions  The number of repetitions of all time "
    echo "                     measurements (for --run), default: 3."
}

# *****************************************************************************
# Argument parsing
# *****************************************************************************

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

runLm=""
runKmeans=""
doCheck=""
doRun=""
repetitions=3

# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            print_help
            exit 0
            ;;
        --lm)
            runLm=1
            ;;
        --kmeans)
            runKmeans=1
            ;;
        --check)
            doCheck=1
            ;;
        --run)
            doRun=1
            ;;
        -r|--repetitions)
            repetitions=$2
            shift
            ;;
        *)
            printf "unknown option: $key\n"
            exit 1
            ;;
    esac
    shift
done

# -----------------------------------------------------------------------------
# Setting some paths
# -----------------------------------------------------------------------------

# Root path of the experiments.
#pathRoot=$(pwd)
pathRoot=$MB_ROOT

#pathArtifacts=$pathRoot/artifacts
pathArtifacts=$MB_DATA_DIR
pathInputs=$pathArtifacts/inputs
#pathResults=$pathArtifacts/results
pathResults=$MB_RESULTS
pathTmp=$pathArtifacts/tmp.txt

mkdir --parents $pathResults
mkdir --parents $pathInputs

# Paths related to DAPHNE.
#pathDaphne=$pathRoot/../prototype
pathDaphne=$DAPHNE_ROOT
daphnec=$pathDaphne/build/bin/daphnec
if [[ ! -d $pathDaphne ]]
then
    echo "Fatal: Expected $pathDaphne to exist. Consider creating a soft"\
         "link to the directory where your DAPHNE prototype resides."
    exit 1
fi

# *****************************************************************************
# Micro benchmark execution
# *****************************************************************************

function convert_args_daphne () {
    if [[ $# -eq 2 ]] # LM
    then
        printf "numRows=$1 numCols=$2"
    elif [[ $# -eq 4 ]] # k-means
    then
        printf "numRecords=$1 numCentroids=$2 numFeatures=$3 numIter=$4"
    fi
}

function convert_inputFiles_daphne () {
    if [[ $# -eq 1 ]] # LM
    then
        printf "inputFile=\"$1\""
    elif [[ $# -eq 2 ]] # k-means
    then
        printf "inputFileX=\"$1\" inputFileC=\"$2\""
    fi
}

function run_daphne () {
    set -e

    local title=$1
    local algo=$2
    local params=$3
    local extraArgs=$4
    local useInputFiles=$5
    local inputFiles=$6
    local outputFile=$7

    local extTitle="DAPHNE ($title)"
    printf "\t\t$extTitle\n"

    cd $pathDaphne

    $daphnec $pathRoot/$algo.daphne \
        --args \
            $(convert_args_daphne $params) \
            useInputFile=$useInputFiles $(convert_inputFiles_daphne $inputFiles) \
        $extraArgs \
        > $outputFile 2> $pathTmp
    printf "$extTitle\t$(echo $params | tr ' ' '\t')\t$repIdx\t$(<$pathTmp)\n" >> $pathRuntime

    cd $pathRoot

    set +e
}

function run_tf () {
    set -e

    local title=$1
    local algo=$2
    local params=$3
    local extraArgs=$4
    local useInputFiles=$5
    local inputFiles=$6
    local outputFile=$7

    local extTitle="TensorFlow ($title)"
    printf "\t\t$extTitle\n"

    $pathRoot/$algo.py $params $extraArgs $useInputFiles $inputFiles \
        > $outputFile 2> $pathTmp
    printf "$extTitle\t$(echo $params | tr ' ' '\t')\t$repIdx\t$(<$pathTmp)\n" >> $pathRuntime

    set +e
}

function run () {
    set -e

    local algo=$1
    local params=$2
    local inputFiles=$3

    echo "Running micro benchmark: $algo for $params"
    
    for repIdx in $(seq $repetitions)
    do
        printf "\trepetition $repIdx\n"
        run_daphne "single-threaded"      $algo "$params"    "" false "$inputFiles" /dev/null
        run_daphne "vectorized pipelines" $algo "$params" --vec false "$inputFiles" /dev/null
        run_tf     "single-threaded"      $algo "$params"     1     0 "$inputFiles" /dev/null
        run_tf     "multi-threaded"       $algo "$params"     0     0 "$inputFiles" /dev/null
    done

    echo "Done."

    set +e
}

# *****************************************************************************
# Correctness checking
# *****************************************************************************

function check_same () {
    cmp --silent $1 $2
    if [[ $? -eq 0 ]]
    then
        echo "ok"
    else
        echo "not ok"
        exit 1
    fi
}

function check () {
    set -e

    local algo=$1
    local params=$2
    local inputFiles=$3

    echo "Checking results: $algo for $params"

    printf "\tRunning\n"

    local pathResDaphneS=$pathResults/resDaphneS.csv
    local pathResDaphneM=$pathResults/resDaphneM.csv
    local pathResTFS=$pathResults/resTFS.csv
    local pathResTFM=$pathResults/resTFM.csv

    run_daphne "single-threaded"      $algo "$params"    "" true "$inputFiles" $pathResDaphneS
    run_daphne "vectorized pipelines" $algo "$params" --vec true "$inputFiles" $pathResDaphneM
    run_tf     "single-threaded"      $algo "$params"     1    1 "$inputFiles" $pathResTFS
    run_tf     "multi-threaded"       $algo "$params"     0    1 "$inputFiles" $pathResTFM

    printf "\tComparing\n"

    printf "\t\tVariants of DAPHNE: "
    check_same $pathResDaphneS $pathResDaphneM

    printf "\t\tVariants of TensorFlow: "
    check_same $pathResTFS $pathResTFM

    printf "\t\tDAPHNE vs. TensofFlow: "
#    local compare=$pathRoot/../scripts/compareFinalResult.py
    local compare=$pathRoot/../p1/compareFinalResult.py
    $compare $pathResDaphneS $pathResTFS

    echo "Done."

    set +e
}

# *****************************************************************************
# Main program
# *****************************************************************************

set -e

# -----------------------------------------------------------------------------
# Setting environment variables
# -----------------------------------------------------------------------------

# Since DAPHNE uses OpenBLAS, but we want to measure DAPHNE's built-in
# multi-threading here.
export OPENBLAS_NUM_THREADS=1

# Such that TensorFlow does not use CUDA/GPU.
export CUDA_VISIBLE_DEVICES=""

# -----------------------------------------------------------------------------
# Creating small test files
# -----------------------------------------------------------------------------
# TODO This should only be necessary for checking correctness.

# For LM.
numRowsCheckLm=50
numColsCheckLm=10
pathInputLm=$pathInputs/rndMatrix.csv

# For k-means.
numRecordsCheckKmeans=50
numCentroidsCheckKmeans=3
numFeaturesCheckKmeans=10
pathInputXKmeans=$pathInputs/rndRecords.csv
pathInputCKmeans=$pathInputs/rndCentroids.csv
pathInputsKMeans="$pathInputXKmeans $pathInputCKmeans"
numIterKmeans=20

printf "Creating test files..."
if [[ $runLm ]]
then
    $pathRoot/rndMatrix.py $numRowsCheckLm $numColsCheckLm $pathInputLm
fi
if [[ $runKmeans ]]
then
    $pathRoot/rndMatrix.py $numRecordsCheckKmeans   $numFeaturesCheckKmeans $pathInputXKmeans
    $pathRoot/rndMatrix.py $numCentroidsCheckKmeans $numFeaturesCheckKmeans $pathInputCKmeans
fi
printf " done.\n"

# -----------------------------------------------------------------------------
# Checking correctness, if requested
# -----------------------------------------------------------------------------

if [[ $doCheck ]]
then
    pathRuntime=/dev/null
    if [[ $runLm ]]
    then
        check lm "$numRowsCheckLm $numColsCheckLm" "$pathInputLm"
    fi
    if [[ $runKmeans ]]
    then
        check kmeans "$numRecordsCheckKmeans $numCentroidsCheckKmeans $numFeaturesCheckKmeans $numIterKmeans" "$pathInputsKMeans"
    fi
fi

# -----------------------------------------------------------------------------
# Running micro benchmarks, if requested
# -----------------------------------------------------------------------------

if [[ $doRun ]]
then
    if [[ $runLm ]]
    then
#        pathRuntime=$pathArtifacts/runtimes_lm.csv
        pathRuntime=$pathResults/runtimes_lm.csv
        printf "system\tnumRows\tnumCols\trepIdx\truntime [ns]\n" > $pathRuntime

        # Toy sizes for testing.
#        run lm    "1000 100" "$pathInputLm"
#        run lm   "10000 100" "$pathInputLm"

        # Sizes for the paper.
        run lm  "100000 1000" "$pathInputLm"
        run lm "1000000 1000" "$pathInputLm"
    fi
    if [[ $runKmeans ]]
    then
#        pathRuntime=$pathArtifacts/runtimes_kmeans.csv
        pathRuntime=$pathResults/runtimes_kmeans.csv
        printf "system\tnumRecords\tnumCentroids\tnumFeatures\tnumIter\trepIdx\truntime [ns]\n" > $pathRuntime

        # Toy sizes for testing.
#        run kmeans    "1000 5 10 $numIterKmeans" "$pathInputsKMeans"
#        run kmeans   "10000 5 10 $numIterKmeans" "$pathInputsKMeans"

        # Sizes for the paper.
        run kmeans   "10000 5 1000 $numIterKmeans" "$pathInputsKMeans"
        run kmeans  "100000 5 1000 $numIterKmeans" "$pathInputsKMeans"
    fi
fi

set +e