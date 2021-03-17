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