/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

lexer grammar DaphneLexer;

KW_WHILE: 'while';
KW_DEF: 'def';
KW_IN: 'in';
KW_IF: 'if';
KW_FOR: 'for';
KW_USE: 'use';
KW_LET: 'let';
KW_RETURN: 'return';
KW_MATRIX: 'matrix';

fragment ALPHABET: [a-zA-Z];

fragment DEC_DIGIT: [0-9];

IDENTIFIER:
	ALPHABET (ALPHABET | DEC_DIGIT | '_')*
	| '_' (ALPHABET | DEC_DIGIT | '_')+;

STRING_LITERAL: '"' (ESCAPE_SEQ | ~["])* '"';

INTEGER_LITERAL: DEC_LITERAL INTEGER_TYPE?;

DEC_LITERAL: DEC_DIGIT (DEC_DIGIT | '_')*;

FLOAT_LITERAL: DEC_LITERAL '.' DEC_LITERAL? FLOAT_TYPE?;

I64: 'i64';
I32: 'i32';
F64: 'f64';
F32: 'f32';
INTEGER_TYPE: I64 | I32;
FLOAT_TYPE: F64 | F32;

STRING_TYPE_LITERAL: 'str';

PLUS: '+';
MINUS: '-';
STAR: '*';
SLASH: '/';
CARET: '^';
EQ: '=';
EQEQ: '==';
NE: '!=';
GT: '>';
LT: '<';
GE: '>=';
LE: '<=';
AT: '@';
UNDERSCORE: '_';
COMMA: ',';
SEMI: ';';
COLON: ':';

LCURLYBRACE: '{';
RCURLYBRACE: '}';
LSQUAREBRACKET: '[';
RSQUAREBRACKET: ']';
LPAREN: '(';
RPAREN: ')';

fragment ESCAPE_SEQ: '\\' [btnfr"'\\0];

LINE_COMMENT: '//' ~[\r\n]* -> skip;

BLOCK_COMMENT: '/*' .*? '*/' -> skip;

WS: [ \t\r\n]+ -> skip;