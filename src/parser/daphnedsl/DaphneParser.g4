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
