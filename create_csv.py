import argparse
import csv
import json
import numpy as np
import pandas as pd
import random
import string

def random_string(length=5):
    return ''.join(random.choices(string.ascii_letters, k=length))

def fixed_str_16():
    # Generate a fixed 16-character string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def generate_column_data(typ, num_rows):
    if typ == "uint8":
        return np.random.randint(0, 256, num_rows, dtype=np.uint8)
    elif typ == "int8":
        return np.random.randint(-128, 128, num_rows, dtype=np.int8)
    elif typ == "uint32":
        return np.random.randint(0, 10000, num_rows, dtype=np.uint32)
    elif typ == "int32":
        return np.random.randint(-1000, 1000, num_rows, dtype=np.int32)
    elif typ == "uint64":
        return np.random.randint(0, 10000, num_rows, dtype=np.uint64)
    elif typ == "int64":
        return np.random.randint(-10000, 10000, num_rows, dtype=np.int64)
    elif typ == "float32":
        return np.random.rand(num_rows).astype(np.float32)
    elif typ == "float64":
        return np.random.rand(num_rows).astype(np.float64)
    elif typ == "str":
        # Note: generating strings is inherently less vectorized.
        return np.array([random_string(random.randint(3, 8)) for _ in range(num_rows)], dtype=str)
    elif typ == "fixedstr16":
        return np.array([fixed_str_16() for _ in range(num_rows)], dtype=str)
    else:
        raise ValueError(f"Unsupported type: {typ}")

def main():
    parser = argparse.ArgumentParser(description="Generate a CSV with variable types in each column.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows")
    parser.add_argument("--cols", type=int, default=7, help="Number of columns")
    parser.add_argument("--output", type=str, default="output.csv", help="Output CSV file name")
    parser.add_argument("--header", action="store_true", help="enable header generation")
    parser.add_argument("--use-str", action="store_true", help="enable string generation")
    args = parser.parse_args()

    # Predefined types to cycle through including additional signed and unsigned integer types.
    if args.use_str:
        col_types = [
            "uint8", "int8",
            "uint32", "int32", "uint64", "int64",
            "float32", "float64",
            "str", "fixedstr16"
        ]
    else:
        col_types = [
        "uint8", "int8",
        "uint32", "int32", "uint64", "int64",
        "float32", "float64"
        ]#, "str", "fixedstr16"]

    # Mapping to convert internal type string to meta file valueType.
    type_mapping = {
        "uint8": "ui8",
        "int8": "si8",
        "uint32": "ui32",
        "int32": "si32",
        "uint64": "ui64",
        "int64": "si64",
        "float32": "f32",
        "float64": "f64",
        "str": "str",
        "fixedstr16": "fixedstr16"
    }

    # Generate each column using vectorized operations.
    data = {}
    schema = []
    for c in range(args.cols):
        typ = col_types[c % len(col_types)]
        col_name = f"col_{c}_{typ}"
        label = col_name if args.header else str(c)
        data[col_name] = generate_column_data(typ, args.rows)
        schema.append({
            "label": label,
            "valueType": type_mapping[typ]
        })

    # Create a DataFrame from the generated data.
    df = pd.DataFrame(data)

    # Write CSV file using pandas which leverages lower-level C code
    df.to_csv(args.output, index=False, header=args.header)
    print(f"CSV file '{args.output}' with {args.rows} rows and {args.cols} columns created.")


    # Create meta data.
    meta = {
        "numRows": args.rows,
        "numCols": args.cols,
        "schema": schema
    }
    meta_filename = f"{args.output}.meta"
    with open(meta_filename, mode="w") as metafile:
        json.dump(meta, metafile, indent=4)
    print(f"Meta file '{meta_filename}' created.")

if __name__ == "__main__":
    main()