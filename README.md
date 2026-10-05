# sql-query-planner

SQL parsing, logical rewriting and cost-based physical planning (Python standard library only).

## Install and entry point

```
python3 -m pip install -e .
sql-query-planner --help
sql-query-planner describe
```

Package `sqlplan`; console script `sql-query-planner`. Every command writes **only JSON to stdout**; a
failure writes **one JSON document to stderr** (so stderr can be parsed as JSON) and exits non-zero.

## The SQL subset

```
SELECT <expr> [AS <alias>] [, ...] | *
FROM <table>
[INNER JOIN <table> ON <left-col> = <right-col>]
[WHERE <condition>]
[GROUP BY <column> [, ...]]
[ORDER BY <column> [ASC|DESC]]
[LIMIT <n>]
```

* one optional `INNER JOIN` of exactly two tables, with a single `ON` equality whose columns come
  one from each input; outer joins, non-equi joins, self joins and three-or-more-table queries are
  rejected as `plan_error`
* columns may be qualified with their table name (`orders.cid`); an unqualified column is valid only
  when unique across the two inputs — a same-name collision, an unknown table or an unknown column is
  a `plan_error`; a malformed `ON` is a `parse_error` carrying `line` and `column`
* aggregates: `count` · `sum` · `min` · `max` · `avg`, each optionally `DISTINCT`; only `count` accepts `*`
* conditions: `= <> < <= > >=`, `IN (...)`, `NOT IN (...)`, `IS NULL`, `IS NOT NULL`, `NOT`, `AND`, `OR`,
  parentheses; `AND` binds tighter than `OR`
* literals: integers, decimals, single-quoted strings (`''` is an escaped quote)
* `--` starts a comment; keywords are case-insensitive, identifiers keep their case

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `describe` | capabilities, operators, statistics source, exit codes, extension surface | 0 |
| `parse --sql <text>` | the AST as JSON | 0 / 2 |
| `plan --sql <text> --table <name=jsonl> … [--index <name.col>]` | logical + physical plan, estimates, applied rewrites | 0 / 2 |
| `explain …` | the same, plus an indented `tree` of the physical plan | 0 / 2 |
| `run … [--output <path>]` | execute and return `{columns, rows, plan}` | 0 / **3** (no rows) / 2 |
| `reconcile … [--output <path>]` | execute through two catalogues (with and without the index) and compare | 0 / **3** (plans disagreed) / 2 |

`--table` is repeatable for a join, in FROM order: `--table orders=orders.jsonl --table
customers=customers.jsonl`. A bare `--table rows.jsonl` still works for a one-table query, and a
bare `--index region` still declares an unqualified index for one table; join indexes must be
qualified (`--index orders.cid`). A missing, duplicated, malformed or unused table binding — or an
index on an unknown table — is a `validation_error`. Any `--output` equal to any input path is
rejected as `output_error` before a single row is read. `--table -` reads that table's rows from
stdin; `--index` is repeatable.

## Input and statistics

Rows are JSON Lines, one object per row. The planner **derives its statistics from the tables
themselves**: row count, and a per-column distinct count. Selectivity follows from those numbers when
available and from documented defaults otherwise (`=` 0.1, range 0.3, `IN` 0.25 per value,
`IS NULL` 0.05). An index scan is chosen **only when it is cheaper**, and the cost constants make
that concrete: `INDEX_COST = 2.0` means an index must save more than two row reads to win. With
three rows and two regions the two plans cost exactly the same, so the full scan is kept — the
planner is not "always prefer the index".

## What the planner promises

* **projection pruning** — each scan reads only the columns that survive to the top from that input
  (`notes` says so)
* **predicate pushdown** — a filter whose column is answered by the index is consumed by the index
  scan; in a join every `WHERE` conjunct that references one input alone is pushed into that
  input's scan, and a conjunct that references both inputs stays as a filter above the join
* **cost-based access path** — `full-scan` or `index-scan`, decided, not assumed
* **cost-based join** — `hash-join` building either side and, when the inner join key is indexed,
  `index-nested-loop-join` are all costed on the *filtered* row counts and distinct estimates; the
  cheapest plan wins, ties preferring hash-join and then the FROM-order build side. The plan JSON and
  the explain tree publish the join operator, its left/right inputs, join keys, `joinOrder`,
  estimated output `rows` and `cost`, so changing the data or adding an index flips an observable
  choice
* **stable join output** — duplicate join keys produce the full cross product, a null key matches
  nothing, and without `ORDER BY` rows come out in logical-left original order then right original
  order for every physical algorithm

`reconcile` is the guarantee that matters: the same statement executed through different physical
plans must produce identical columns and rows. It reports `identical`, both plans' `accessPath`,
`joinOperator`, `joinOrder`, their estimated costs, and — when they disagree — both result sets, and
exits **3**.

## Guarantees

* Unspecified `--output` writes to stdout; otherwise the result is written to a temporary file and
  atomically replaced, so a failure never corrupts the previous file or leaves a partial one.
* An output path equal to any table path is rejected as `output_error` **before** anything is read.
* Every error document is `{"error":"<kind>","message":"…", …}`; parse errors carry `line` and `column`,
  row-loading errors carry `line`.

## Layout

```
sqlplan/lexer.py     tokeniser with 1-based positions, comments and string escapes
sqlplan/parser.py    recursive-descent parser and the AST (with to_document), including INNER JOIN
sqlplan/planner.py   logical nodes, rewrites, selectivity, access-path and join cost models, explain
sqlplan/executor.py  scans, filter, hash/index-nested-loop joins, aggregate, project, sort, limit
sqlplan/cli.py       six subcommands and the exit-code contract
tests/               parser positions, plan choices, join planning and execution, CLI surface
```
