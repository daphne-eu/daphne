<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DaphneDSL

DaphneDSL is DAPHNE's domain-specific language (DSL).
DaphneDSL is written in plain text files, typically ending with `.daphne`.
It is a case-sensitive language inspired by ML systems as well as
languages and libraries for numerical computation like Julia, Python NumPy,
R, and SystemDS DML.
Its syntax is inspired by C/Java-like languages.

## Hello World

A simple hello-world script can look as follows:

```
print("hello world");
```

Assuming this script is stored in the file `hello.daphne`, it can be executed by the following command:

```
build/bin/daphne hello.daphne
```

The remainder of this document discusses in detail how to write a DaphneDSL script and how to run it.


## Writing a DaphneDSL script

This section presents the various language features of DaphneDSL *as they are right now*, but *note that DaphneDSL is still evolving*.
Furthermore, the reader is assumed to be familiar with programming in general.

### Variables

Variables are used to refer to values.
Valid identifiers start with a letter (`a-z`, `A-Z`) or an underscore (`_`) that can be followed by any number of letters (`a-z`, `A-Z`), underscores (`_`), and decimal digits (`0-9`).

*Examples*
```
X
y
_hello123
a_long_Variable123_456NAME
```

Variables do not need to be (and cannot be) declared.
Instead, simply assign a value to a variable and its type will be inferred.
Variables must have been assigned to before they are used in an expression.

### Types

DaphneDSL differentiates *data types* and *value types*.

Currently, DaphneDSL supports the following *abstract* data types:
- *matrix*: homogeneous value type
- *frame*: a table with columns of potentially different value types
- *scalar*: a single value

The currently supported value types are:
- floating-point numbers of various widths: `f64`, `f32`
- signed and unsigned integers of various widths: `si64`, `si32`, `si8`, `ui64`, `ui32`, `ui8`
- booleans `bool` and strings `str` *(currently only for scalars)*

### Comments

DaphneDSL supports single-line comments (starting with `#` or `//`) and multi-line comments (everything enclosed in `/*` and `*/`).

*Examples*
```
# this is a comment
print("Hello World!"); // this is also a comment
/* comments can
span multiple
lines */
```

### Expressions

#### Simple Expressions

Simple expressions constitute the basis of all expressions, and DaphneDSL knows three kinds:

##### Literals

Literals represent hard-coded values and can be of different types:

**Integer literals** are specified in decimal notation and have the type `si64`.

*Examples*
```
0
123
-456
```

**Floating-point literals** are specified in decimal notation and have the type `f64`.
Furthermore, the following literals stand for special floating-point values: `nan`, `inf`, `-inf`.

*Examples*
```
0.0
123.0
-456.78
inf
nan
```

**Boolean literals** can be `false` and `true`.

*Examples*
```
false
true
```

**String literals** are enclosed in quotation marks `"`.
Special characters must be escaped using a backslash:
- `\n`: new line
- `\t`: tab
- `\"`: quotation mark
- `\\`: backslash
- `\b`: backspace
- `\f`: line feed
- `\r`: carriage return

*Examples*
```
"Hello World!"
"line 1\nline 2\nline 3"
"This is \"hello.daphne\"."
```

##### Variables

Variables are referenced by their name.

*Examples*
```
x
```

##### Script arguments

Script arguments are named *literals* that can be passed to a DaphneDSL script.
They are referenced by a dollar sign `$` followed by the argument's name.

*Examples*
```
$x
```

#### Complex Expressions

DaphneDSL offeres several ways to build more complex expressions.

##### Operators

DaphneDSL currently supports the following binary operators:

| Operator | Meaning |
| --- | --- |
| `@` | matrix multiplication (highest precedence) |
| `^` | exponentiation |
| `%` | modulo |
| `*`, `/` | multiplication, division |
| `+`, `-` | addition/string concatenation, subtraction |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | comparison |
| `&&` | logical AND |
| `\|\|` | logical OR (lowest precedence) |

*We plan to add more operators here, including unary operators.*

*Matrix multiplication (`@`):*
The inputs must be matrices of compatible shapes, and the output is always a matrix.

*All other operators:*
The following table shows which combinations of inputs are allowed and which result they yield:

| Left input | Right input | Result | Details |
| --- | --- | --- | --- |
| scalar | scalar | scalar | binary operation of two scalars |
| matrix (n x m) | scalar | matrix (n x m) | element-wise operation of each value with scalar |
| matrix (n x m) | matrix (n x m) | matrix (n x m) | element-wise operation on corresponding values |
| matrix (n x m) | matrix (1 x m) | matrix (n x m) | broadcasting of row-vector |
| matrix (n x m) | matrix (n x 1) | matrix (n x m) | broadcasting of column-vector |

*Examples*
```
1.5 * X @ y + 0.001
x == 1 && y < 3.5
```

##### Parantheses

Parantheses can be used to manually control operator precedence.

*Examples*
```
1 * (2 + 3)
```

##### Indexing

(Right) indexing enables the extraction of a part of the rows and/or columns of a data object (matrix/frame) into a new data object.
The result is always a data object of the same type as the input (even *1 x 1* results need to be casted to scalars explicitly).

The rows and columns to extract can be specified independently in any of the following ways:

**Omit indexing**

Omiting the specification of rows/columns means extracting all rows/columns.

*Examples*
```
X[, ] # same as X (all rows and columns)
```

**Indexing by position**

This is supported for addressing rows and columns in matrices and frames.

- *Single row/column position:*
  Extracts only the specified row/column.

  *Examples*
  ```
  X[2, 3] # extracts the cell in row 2, column 3 as a 1 x 1 matrix
  ```

- *Row/column range:*
  Extracts all rows/columns between a lower bound (inclusive) and an upper bound (exclusive).
  The lower and upper bounds can be omited independently of each other.
  In that case, they are replaced by zero and the number of rows/columns, respectively.
  
  *Examples*
  ```
  X[2:5, 3] # extracts rows 2, 3, 4 of column 3
  X[2, 3:]  # extracts row 2 of all columns from column 3 onwards
  X[:5, 3]  # extracts rows 0, 1, 2, 3, 4 of column 3
  X[:, 3]   # extracts all rows of column 3
  ```

- *Arbitrary sequence of row/column positions:*
  Expects a sequence of row/column positions *as a column (n x 1) matrix*.
  There are no restrictions on these positions, except that they must be in bounds.
  In particular, they do *not* need to be contiguous, sorted, or unique.

  *Examples*
  ```
  pos1 = seq(5, 1, -2); # [5, 3, 1]
  X[pos1, ]             # extracts rows 5, 3, 1 of all columns

  pos2 = fill(2, 3, 1); # [2, 2, 2]
  X[, pos2]             # extracts column 2 three times
  ```

A few remarks on positions:
- Counting starts at zero.
  For instance, a 5 x 3 matrix has row positions 0, 1, 2, 3, and 4, and column positions from 0, 1, and 2.
- They must be non-negative.
- They can be provided as integers or floating-point numbers (the latter are rounded down to integers).
- They can be given as literals or as any expression evaluating to a suitable value.

*Examples*
```
X[1.2, ]              # same as X[1, ]
X[1.9, ]              # same as X[1, ]
X[i, (j + 2*sum(Y)):] # expressions
```

**Indexing by label**

So far, this is only supported for addressing columns of frames.

- *Single column label:*
  Extracts only the column with the given label.

  *Examples*
  ```
  X[, "revenue"]        # extracts the column labeled "revenue"
  X[100:200, "revenue"] # extracts rows 100 through 199 of the column labeled "revenue"
  ```

**Indexing by bit vector (filtering)**

So far, this is only supported for addressing rows of frames.

In contrast to indexing with positions and labels, indexing by bit vector specifies for each row whether to retain it or not.
It expects a column (*n x 1*) matrix with as many rows as the input data object.
"Bit" vector is meant in a conceptual sense, the actual value type can be any integer or floating-point type, but all entries must be either zero or one.
- If the *i*-th entry of the bit vector is zero, the *i*-th row of the input is skipped
- If the *i*-th entry of the bit vector is one, the *i*-th row of the input is retained

*Examples*
```
bv = seq(1, 5, 1) <= 3; # [1, 1, 1, 0, 0]
X[[bv, ]]               # extracts rows 0, 1, 2
```

Note that double square brackets (`[[...]]`) must be used to distinguish indexing by bit vector from indexing by an arbitrary sequence of positions.
Furthermore, the specification of columns must be omited here.

##### Casts

##### Function calls

### Statements

#### Assignment

#### Block

#### Control Flow

##### If-then-else

##### Loops

**For-Loops**

**While-Loops**

**Do-While-Loops**

### Functions

### Selected Aspects

#### Using SQL inside DaphneDSL

#### File I/O

### Planned Features

## Running a DaphneDSL script