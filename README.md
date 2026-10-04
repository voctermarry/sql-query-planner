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
[WHERE <condition>]
[GROUP BY <column> [, ...]]
[ORDER BY <column> [ASC|DESC]]
[LIMIT <n>]
```

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
| `plan --sql <text> --table <jsonl> [--index <column>]` | logical + physical plan, estimates, applied rewrites | 0 / 2 |
| `explain …` | the same, plus an indented `tree` of the physical plan | 0 / 2 |
| `run … [--output <path>]` | execute and return `{columns, rows, plan}` | 0 / **3** (no rows) / 2 |
| `reconcile … [--output <path>]` | execute through two catalogues (with and without the index) and compare | 0 / **3** (plans disagreed) / 2 |

`--table -` reads rows from stdin. `--index` is repeatable.

## Input and statistics

Rows are JSON Lines, one object per row. The planner **derives its statistics from the table itself**:
row count, and a per-column distinct count. Selectivity follows from those numbers when available and
from documented defaults otherwise (`=` 0.1, range 0.3, `IN` 0.25 per value, `IS NULL` 0.05). An index
scan is chosen **only when it is cheaper**, and the cost constants make that concrete:
`INDEX_COST = 2.0` means an index must save more than two row reads to win. With three rows and two
regions the two plans cost exactly the same, so the full scan is kept — the planner is not "always
prefer the index".

## What the planner promises

* **projection pruning** — the scan reads only the columns that survive to the top (`notes` says so)
* **predicate pushdown** — a filter whose column is answered by the index is consumed by the index scan;
  the remaining conjuncts are listed as residual predicates
* **cost-based access path** — `full-scan` or `index-scan`, decided, not assumed
* **explicit extension surface** — `describe` states what this build does *not* plan (joins,
  index-only scans, block indexes), so the next task has a declared boundary instead of a guess

`reconcile` is the guarantee that matters: the same statement executed through different physical plans
must produce identical columns and rows. It reports `identical`, both plans' `accessPath`, their
estimated costs, and — when they disagree — both result sets, and exits **3**.

## Guarantees

* Unspecified `--output` writes to stdout; otherwise the result is written to a temporary file and
  atomically replaced, so a failure never corrupts the previous file or leaves a partial one.
* An output path equal to the table path is rejected as `output_error` **before** anything is read.
* Every error document is `{"error":"<kind>","message":"…", …}`; parse errors carry `line` and `column`,
  row-loading errors carry `line`.

## Layout

```
sqlplan/lexer.py     tokeniser with 1-based positions, comments and string escapes
sqlplan/parser.py    recursive-descent parser and the AST (with to_document)
sqlplan/planner.py   logical nodes, rewrites, selectivity and cost model, catalogue, explain
sqlplan/executor.py  physical execution: scans, filter, aggregate, project, sort, limit
sqlplan/cli.py       six subcommands and the exit-code contract
tests/               parser positions, plan choices, aggregation semantics, CLI surface
```
