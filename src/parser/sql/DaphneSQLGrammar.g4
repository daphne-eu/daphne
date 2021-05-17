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

grammar DaphneSQLGrammar;

// ****************************************************************************
// Parser rules
// ****************************************************************************
script:
    query* EOF ;

query:
  subquery? select';';

select:
  SQL_SELECT (SQL_ALL | SQL_DISTINCT | SQL_UNIQUE)? select_list
  SQL_FROM table_list
/*  where_clause?//*/
/*
  group_by_clause?
  (SQL_HAVING condition)?
  ( (SQL_UNION SQL_ALL? | SQL_INTERSECT | SQL_MINUS) ( subquery ))?
  order_by_clause?
*/
;
/* query_name is a temporary identifier */
subquery:
  SQL_WITH subquery_list;

subquery_list:
  alias SQL_AS '(' select ')' (',' subquery_list)*;

/* Needs to be extended. For instance selecting everything from a table*/
select_list:
  '*'
  | expr (SQL_AS alias)? (',' select_list)?;

table_list:
  table_reference (',' table_list | join_list)?;

/* ONLY keyword and flashback clause excluded for the moment*/
table_reference:
  query_table_expression (SQL_AS? alias)?;

query_table_expression:
  (IDENTIFIER '.')? IDENTIFIER;

expr:
  IDENTIFIER ('.' IDENTIFIER | '.' INT_POSITIV_LITERAL)?;

alias:
  IDENTIFIER;

condition:
  expr
  | condition '&&' condition
  | expr '=' expr
  ;

join_list:
  join_clause join_list?;

join_clause:
  inner_cross_join_clause | outer_join_clause;

inner_cross_join_clause:
  SQL_INNER? SQL_JOIN table_reference join_condition
  | (SQL_CROSS | SQL_NATURAL SQL_INNER?) SQL_JOIN table_reference
  ;

outer_join_clause:
  SQL_NATURAL? outer_join_type SQL_JOIN table_reference join_condition;

join_condition:
  SQL_ON condition
  | SQL_USING '(' expr ')'
  ;

outer_join_type:
  (SQL_FULL | SQL_LEFT | SQL_RIGHT) SQL_OUTER?;

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

INT_LITERAL:
    ('0' | '-'? NON_ZERO_DIGIT DIGIT*) ;

INT_POSITIV_LITERAL:
    ('0' | NON_ZERO_DIGIT DIGIT*) ;

FLOAT_LITERAL:
    '-'? (DIGIT* '.' DIGIT+ | DIGIT+ '.' DIGIT*);

WS: [ \t\r\n]+ -> skip;
