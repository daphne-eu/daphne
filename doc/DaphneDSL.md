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
| `+`, `-` | addition, subtraction |
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