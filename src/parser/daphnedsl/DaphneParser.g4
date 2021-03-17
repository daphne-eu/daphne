parser grammar DaphneParser;

options {
	tokenVocab = DaphneLexer;
}

file: item* EOF;

item: function;

function:
	KW_DEF IDENTIFIER '(' functionArgs? ')' blockStatement;

functionArgs: functionArg (',' functionArg)* ','?;

functionArg: IDENTIFIER ':' type;

type: INTEGER_TYPE # IntegerType
 | FLOAT_TYPE # FloatType
 | 'str' # StringType;

blockStatement: '{' statement* '}';

statement:
	blockStatement
	| expressionStatement
	| letStatement
	| whileStatement;

expressionStatement: expression ';';

expression:
	literalExpressionRule												# LiteralExpression
	| IDENTIFIER													# IdentifierExpression
	| fn=IDENTIFIER '(' parameters? ')'								# CallExpression
	//| expression '[' expression ']'									# IndexExpression
	//| MINUS expression											# NegationExpression
	| lhs=expression op='@' rhs=expression			# ArithmeticExpression
	| lhs=expression op=('*' | '/') rhs=expression			# ArithmeticExpression
	| lhs=expression op=('+' | '-') rhs=expression			# ArithmeticExpression
	| lhs=expression op=('==' | '!=' | '<' | '>' | '<=' | '>=') rhs=expression	# ComparisonExpression
	| IDENTIFIER '=' expression										# AssignmentExpression
	| '(' expression ')'											# GroupedExpression
	| 'return' expression?											# ReturnExpression;


parameters: parameter (',' parameter)* ','?;

parameter: expression;

letStatement: KW_LET IDENTIFIER '=' expression ';';

whileStatement: KW_WHILE '(' expression ')' blockStatement;

literalExpressionRule:
	STRING_LITERAL
	| INTEGER_LITERAL
	| FLOAT_LITERAL
	| matrixLiteral;

matrixLiteral:
    '[' matrixLiteralElements ']';

matrixLiteralElements: literalExpressionRule (',' literalExpressionRule)* ','?;
