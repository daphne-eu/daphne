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

Currently, DaphneDSL supports the following *abstract* **data types**:
- *matrix*: homogeneous value type for all cells
- *frame*: a table with columns of potentially different value types // individual value type for each column
- *scalar*: a single value

The currently supported **value types** are:
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

*Examples*: `0`, `123`, `-456`

**Floating-point literals** are specified in decimal notation and have the type `f64`.
Furthermore, the following literals stand for special floating-point values: `nan`, `inf`, `-inf`.

*Examples*: `0.0`, `123.0`, `-456.78`, `inf`, `nan`

**Boolean literals** can be `false` and `true`.

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

DaphneDSL offers several ways to build more complex expressions.

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

*We plan to add more operators, including unary operators.*

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
  X[:, 3]   # extracts all rows of column 3, same as X[, 3]
  ```

- *Arbitrary sequence of row/column positions:*
  Expects a sequence of row/column positions *as a column (n x 1) matrix*.
  There are no restrictions on these positions, except that they must be in bounds.
  In particular, they do *not* need to be contiguous, sorted, or unique.

  *Examples*
  ```
  pos1 = seq(5, 1, -2); # [5, 3, 1]
  X[pos1, ]             # extracts rows 5, 3, and 1

  pos2 = fill(2, 3, 1); # [2, 2, 2]
  X[, pos2]             # extracts column 2 three times
  ```

A few remarks on positions:
- Counting starts at zero.
  For instance, a 5 x 3 matrix has row positions 0, 1, 2, 3, and 4, and column positions 0, 1, and 2.
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

Values can be casted to a particular type explicitly.
Currently, it is possible to cast:
- between scalars of different types
- between matrices of different value types
- from matrix to frame
- from frame to matrix

*Examples*
```
as.f32(x)        # casts x to f32 scalar
as.ui8(x)        # casts x to ui8 scalar
as.matrix.f32(x) # casts x to matrix with value type f32
as.matrix(x)     # casts x to matrix with value type infered from x
as.frame(x)      # casts x to frame
```

TODO: We should have a scalar data type for casts (with specified or infered value type).

TODO: Elaborate on casts between scalars and data objects, once we support that.

##### Function calls

Function calls can address *built-in* functions as well as *user-defined* functions, but the syntax is the same in both cases:
The name of the function followed by a comma-separated list of positional parameters in parantheses.

TODO: Create separate reference of all built-in functions and DaphneDSL standard lib functions and link them here.

TODO: Elaborate on types of given vs expected args.

*Examples*
```
print("hello");
t(myMatrix);
seq(0, 10, 2);
```

### Statements

At the highest level, a DaphneDSL script is a sequence of statements.
Statements comprise assignments, various forms of control flow, and declarations of user-defined functions.

#### Expression statement

Every expression followed by a semicolon `;` can be used as a statement.
This is useful for expressions (especially function calls) which do not return a value.
Nevertheless, it can also be used for expressions with one or more return values, in which case these values are ignored.

*Examples*
```
print("hello"); # built-in function without return value
1 + 2;          # value is ignored, useless but possible
doSomething();  # possible return values are ignored, but the execution 
                # of the user-defined function could have side effects
```

#### Assignment statement

The return value(s) of an expression can be assigned to one (or more) variable(s).

**Single-assignments** are used for expressions with exactly one return value.

*Examples*
```
x = 1 + 2;
```

**Multi-assignments** are used for expressions with more than one return value.

*Examples*
```
evals, evecs = eigen(A); # eigen() returns two values, the (n x 1)-matrix of
                         # eigen-values and the (n x n)-matrix of eigen-vectors
                         # of the input matrix A.
```

##### Indexing

The value of an expression can also be assigned to a *partition* of an *existing data object*.
This is done by (left) indexing, whose syntax is similar to (right) indexing in expressions.

Currently, left indexing is supported only for matrices.
Furthermore, the rows/columns cannot be addressed by arbitrary positions lists or bit vectors (yet).

The following conditions must be fulfilled:
- The left-hand-side variable must have been initialized.
- The left-hand-side variable must represent a matrix.
- The right-hand-side expression must return a matrix.
- The shapes of the partition addressed on the left-hand side and the return value of the right-hand-side expression must match.
- The value type of the left-hand-side and right-hand-side matrices must match.

*Examples*
```
X[5, 2] = fill(123, 1, 1);        # insert (1 x 1)-matrix
X[10:20, 2:5] = fill(123, 10, 3); # insert (10 x 3)-matrix
```

Left indexing can be used with both single and multi-assignments.
With the latter, it can be used with each variable on the left-hand side individually and independently.

*Examples*
```
x, Y[3, :], Z = calculateSomething();
```

**Copy-on-write semantics**

Left indexing enables the modification of existing data objects, whereby the semantics is *copy-on-write*.
That is, if two different variables represent the same runtime data object, then left indexing on one of these variables does not have any effects on the others.
This is achieved by transparently copying the data as necessary.

*Examples*
```
A = ...;           # some matrix
B = A;             # copy-by-reference
B[..., ...] = ...; # copy-on-write: changes B, but no effect on A
A[..., ...] = ...; # copy-on-write: changes A, but no effect on B
```

#### Block statement

A block statement allows to view an enclosed sequence of statements like a single statement.
This is very useful in combination with the control flow statements described below.
Besides that, a block statement starts a new scope in terms of visibility of variables.
Within a block, all variables from outside the block can be read and written.
However, variables created inside a block are not visible anymore after the block.

The syntax of a block statement is:
```
{
    statement1
    statement2
    ...
}
```

*Examples*
```
x = 1;
{
    print(x); # read access
    x = 2;    # write access
    y = 1;    # variable created inside the block
}
print(x);     # prints 2
print(y);     # error
```

#### Control Flow statements

TODO: Maybe move block statement here, for consistency as ease of explanation.
TODO: Mention that control flow statements (including block statements) can be nested arbitrarily.
TODO: Indentations are optional, but are recommended for readability.

##### If-then-else

The syntax of an if-then-else statement is as follows:
```
if (condition)
    then-statement
else
    else-statement
```
*condition* is an expression returning a single value.
If this value is `true` (when casted to value type `bool`, if necessary), the *then-statement* is executed.
Otherwise, the *else-statement* is executed, *if it is present*.
Note that the *else*-branch (keyword and statement) may be omitted.
Furthermore, *then-statement* and *else-statement* can be block statements, to allow any number of statements in the then and else-branches.

*Examples:*
```
if (sum(X) == 0)
    X = X + 1;
```
```
if (2 * x > y) {
    z = z / 2;
    a = true;
}
else
    z = z * 2;
```
```
if (a)
    print("a");
else if (b)
    print("not a, but b");
else
    print("neither a nor b");
```

##### Loops

**For-Loops**

For-loops are used to iterate over the elements of a sequence of integers.
The syntax of a for-loop is as follows:
```
for (var in start:end[:step])
    body-statement
```
*var* must be a valid identifier and is assigned the values from *start* to *end* in increments of *step*.
*start*, *end*, and *step* are expressions evaluating to a single number.
*step* is optional and defaults to 1 if *end* is greater than *start*, or -1 otherwise.
In that sense, for-loops can also be used to count backwards by setting *start* greater than *end*.
The *body-statement* is executed for each value in the sequence, and within the *body-statement, this value is accessible via the read-only variable `var`.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*
```
for(i in 1:3)
    print(i); # 1 2 3
```
```
x = 0; y = 0;
for(i in 10:1:-3) {
    x = x + i;
    y = y + 1;
}
print(x); # 22
print(y); #  4
```

**While-Loops**

While loops are used to execute a (block of) statement(s) as long as an arbitrary condition holds true.
The syntax of a while-loop is as follows:
```
while (condition)
    body-statement
```
*condition* is an expression returning a single value, and is evaluated before each iteration.
If this value is `true` (when casted to value type `bool`, if necessary), the *body-statement* is executed, and the loop starts anew.
Otherwise, the program continues after the loop.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*
```
i = 0;
while(i < 10 && !converged) {
    A = A @ B;
    converged = sum(A);
    i = i + 1;
}
```

**Do-While-Loops**

Do-while-loops are a variant of while-loops, which checks the condition after each iteration.
Consequently, a do-while-loop always executes at least one iteration.
The syntax of a do-while-loop is as follows:
```
do
    body-statement
while (condition);
```
The semicolon at the end is optional.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*
```
i = 5;
do {
    A = sqrt(A);
    i = i - 1;
} while (mean(A) > 100 && i > 0);
```

### Functions

### Selected Aspects

#### Using SQL inside DaphneDSL

#### File I/O

### Planned Features

## Running a DaphneDSL script
