# *****************************************************************************
# Argument parsing
# *****************************************************************************

# -----------------------------------------------------------------------------
# Defaults.
# -----------------------------------------------------------------------------

scaleFactor=1
pathDBGen=./StarSchemaBenchmark

# -----------------------------------------------------------------------------
# Parsing.
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -sf|--scaleFactor)
            scaleFactor=$2
            shift
            ;;
        *)
            printf "unknown option: $key\n"
            exit -1
            ;;
    esac
    shift
done

function createData () {
	set -e

	cd $pathDBGen
	make
	./dbgen -f -s $scaleFactor -T a
	cd ../

	pathData=../../data/ssb/sf$scaleFactor
	mkdir -p $pathData
	python3 ./dbdict.py ./schema_full.json ./schema_required.json $pathDBGen $pathData

	rm -f $pathDBGen/*.tbl
	
	cd $pathData/tbls_dict
	
	for file in *.tbl; do
		mv -- "$file" "../${file%.tbl}.csv"
	done
	
	cd ..
	
	for file in *.csv; do
		sed -i "s|\||,|g" "$file"
	done

	rm -r tbls_dict/

  pwd
	cd ../../../D7.3/data_generation/meta
	pwd
	
	for file in *.meta; do
	   cp  "$file" "../$pathData/$file"
	done
}

createData