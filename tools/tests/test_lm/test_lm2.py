import os
import subprocess
import numpy as np
import json

# Set paths
dml_file = os.path.abspath("../../../thirdparty/systemds/scripts/builtin/lm.dml")
dml_file_caller = os.path.abspath("lm_.dml")

translator_file = os.path.abspath("../../dml2daph.py")
daph_file = os.path.abspath("../../translated_files/lm.daph")   # is generated by translator
daph_file_caller = os.path.abspath("lm_.daph")

# Generate data
icpt = 0
reg = 0.0000001
tol = 0.0000001
maxi = 0

n_samples = 13000
n_features = 1300

np.random.seed(0)
test_matrix = np.random.randn(n_samples, n_features)
test_matrix2 = np.random.randn(n_samples, 1)

os.makedirs(os.path.abspath("data"), exist_ok=True)

test_matrix_file_dml = os.path.abspath("data/test_matrix_dml.csv")
test_matrix_file_daph = os.path.abspath("data/test_matrix_daph.csv")

test_matrix_file_dml2 = os.path.abspath("data/test_matrix_dml2.csv")
test_matrix_file_daph2 = os.path.abspath("data/test_matrix_daph2.csv")

# Save the matrix to a file
np.savetxt(test_matrix_file_dml, test_matrix, delimiter=',')
np.savetxt(test_matrix_file_daph, test_matrix, delimiter=',')
np.savetxt(test_matrix_file_dml2, test_matrix2, delimiter=',')
np.savetxt(test_matrix_file_daph2, test_matrix2, delimiter=',')

# Create metadata for the test matrix
test_matrix_metadata_dml = {
    "data_type": "matrix",
    "format": "csv",
    "rows": test_matrix.shape[0],
    "cols": test_matrix.shape[1]
}

test_matrix_metadata_daph = {
    "numRows": test_matrix.shape[0],
    "numCols": test_matrix.shape[1],
    "valueType": "f64",
    "numNonZeros": 0
}

test_matrix_metadata_dml2 = {
    "data_type": "matrix",
    "format": "csv",
    "rows": test_matrix2.shape[0],
    "cols": test_matrix2.shape[1]
}

test_matrix_metadata_daph2 = {
    "numRows": test_matrix2.shape[0],
    "numCols": test_matrix2.shape[1],
    "valueType": "f64",
    "numNonZeros": 0
}

# Save metadata to a file
with open(test_matrix_file_dml + ".mtd", 'w') as f:
    json.dump(test_matrix_metadata_dml, f)

with open(test_matrix_file_daph + ".meta", 'w') as f:
    json.dump(test_matrix_metadata_daph, f)

with open(test_matrix_file_dml2 + ".mtd", 'w') as f:
    json.dump(test_matrix_metadata_dml2, f)

with open(test_matrix_file_daph2 + ".meta", 'w') as f:
    json.dump(test_matrix_metadata_daph2, f)

# Run the DML script
dml_output_file = os.path.abspath("output/dml_output.csv")
dml_command = "cd ../../../thirdparty/systemds/target && spark-submit SystemDS.jar -f {} -args {} {}".format(dml_file_caller, test_matrix_file_dml, test_matrix_file_dml2)
subprocess.call(dml_command, shell=True)

# Load DML results
dml_results = np.loadtxt(dml_output_file, delimiter=' ')

# Translate DML to Daphne
translate_command = "python3 {} {}".format(translator_file, dml_file)
subprocess.call(translate_command, shell=True)

# Run the Daphne script
daphne_output_file = os.path.abspath("output/daphne_output.csv")
daphne_command = "../../../bin/daphne {} icpt={} reg={} tol={} maxi={} verbose=false".format(daph_file_caller, icpt, reg, tol, maxi)
subprocess.call(daphne_command, shell=True)

# Load Daphne results
daphne_results = np.loadtxt(daphne_output_file, delimiter=',')

# Compare results
print("Comparison for m_lm: {}".format(np.allclose(dml_results[:,2], daphne_results, atol=1e-02)))

