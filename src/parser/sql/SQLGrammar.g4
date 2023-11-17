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

// ****************************************************************************
// Grammar
// ****************************************************************************

grammar SQLGrammar;

// ****************************************************************************
// Parser rules
// ****************************************************************************
script:
    query* EOF ;

sql:
    query  EOF;

query:
    select ';'?;

select:
    SQL_SELECT distinctExpr? selectExpr (',' selectExpr)*
    SQL_FROM tableExpr
    whereClause?
    groupByClause?
    orderByClause?
    ;

subquery:
    SQL_WITH subqueryExpr (',' subqueryExpr)*;

subqueryExpr:
    var=IDENTIFIER SQL_AS '(' select ')';

selectExpr:
    var=generalExpr (SQL_AS aka=IDENTIFIER)?;

tableExpr:
    fromExpr joinExpr*;

distinctExpr:
    SQL_DISTINCT;

fromExpr:
    var=tableReference #tableIdentifierExpr
    | lhs=tableReference ',' rhs=fromExpr #cartesianExpr
    ;

joinExpr:
    SQL_INNER? SQL_JOIN var=tableReference
        SQL_ON rhs=selectIdent op=CMP_OP lhs=selectIdent
        (SQL_AND selectIdent (CMP_OP)? selectIdent)*
        #innerJoin
    ;

whereClause:
    SQL_WHERE cond=generalExpr;

groupByClause:
    SQL_GROUP SQL_BY selectIdent (',' selectIdent)*
    havingClause?;

havingClause:
    SQL_HAVING cond=generalExpr;

orderByClause:
    SQL_ORDER SQL_BY selectIdent orderInformation
    (',' selectIdent orderInformation)*;

orderInformation:
    (asc=SQL_ASC|desc=SQL_DESC)?;

generalExpr:
    literal # literalExpr
    | '*' # starExpr
    | selectIdent # identifierExpr
    | func=IDENTIFIER '(' var=generalExpr ')' #groupAggExpr
    | '(' generalExpr ')' # paranthesesExpr
    | lhs=generalExpr op=('*'|'/') rhs=generalExpr # mulExpr
    | lhs=generalExpr op=('+'|'-') rhs=generalExpr # addExpr
    | lhs=generalExpr op=CMP_OP rhs=generalExpr # cmpExpr
    | lhs=generalExpr SQL_AND rhs=generalExpr # andExpr
    | lhs=generalExpr SQL_OR rhs=generalExpr # orExpr
    ;

tableReference:
    var=IDENTIFIER (SQL_AS? aka=IDENTIFIER)?;

selectIdent:
    (frame=IDENTIFIER '.')? var=(IDENTIFIER|'*')  #stringIdent
    ;

literal:
    INT_LITERAL
    | FLOAT_LITERAL
    ;

// ****************************************************************************
// Lexer rules
// ****************************************************************************

SQL_SELECT: S E L E C T;
SQL_FROM: F R O M;
SQL_ALL: A L L;
SQL_DISTINCT: D I S T I N C T;
SQL_UNIQUE: U N I Q U E;
SQL_HAVING: H A V I N G;
SQL_UNION: U N I O N;
SQL_INTERSECT: I N T E R S E C T;
SQL_MINUS: M I N U S;
SQL_WITH: W I T H;
SQL_AS: A S;
SQL_ONLY: O N L Y;
SQL_INNER: I N N E R;
SQL_JOIN: J O I N;
SQL_ON: O N;
SQL_USING: U S I N G;
SQL_CROSS: C R O S S;
SQL_NATURAL: N A T U R A L;
SQL_FULL: F U L L;
SQL_LEFT: L E F T;
SQL_RIGHT: R I G H T;
SQL_OUTER: O U T E R;
SQL_WHERE: W H E R E;
SQL_GROUP: G R O U P;
SQL_BY: B Y;
SQL_ORDER: O R D E R;
SQL_ASC: A S C;
SQL_DESC: D E S C;
SQL_AND: A N D;
SQL_OR: O R;

fragment A: [aA];
fragment B: [bB];
fragment C: [cC];
fragment D: [dD];
fragment E: [eE];
fragment F: [fF];
fragment G: [gG];
fragment H: [hH];
fragment I: [iI];
fragment J: [jJ];
fragment K: [kK];
fragment L: [lL];
fragment M: [mM];
fragment N: [nN];
fragment O: [oO];
fragment P: [pP];
fragment Q: [qQ];
fragment R: [rR];
fragment S: [sS];
fragment T: [tT];
fragment U: [uU];
fragment V: [vV];
fragment W: [wW];
fragment X: [xX];
fragment Y: [yY];
fragment Z: [zZ];

fragment LETTER: [a-zA-Z];
fragment DIGIT: [0-9];
fragment NON_ZERO_DIGIT: [1-9];

IDENTIFIER:
    (LETTER | '_')(LETTER | '_' | DIGIT)* ;

CMP_OP:
    ('='|'<>'|'!='|'<='|'>='|'<'|'>');

INT_LITERAL:
    '-'? DIGIT+;

FLOAT_LITERAL:
    '-'? ( DIGIT+ '.' DIGIT+);

WS: [ \t\r\n]+ -> skip;
