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
FROM <table> [INNER JOIN <table> ON <column> = <column>]
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
* columns may be table-qualified (`orders.id`); inside a join an unqualified column must exist in
  exactly one input, otherwise the query fails with `plan_error` (`ambiguous column`)

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `describe` | capabilities, operators, statistics source, exit codes, extension surface | 0 |
| `parse --sql <text>` | the AST as JSON | 0 / 2 |
| `plan --sql <text> --table <jsonl> [--index <column>]` | logical + physical plan, estimates, applied rewrites | 0 / 2 |
| `explain …` | the same, plus an indented `tree` of the physical plan | 0 / 2 |
| `run … [--output <path>]` | execute and return `{columns, rows, plan}` | 0 / **3** (no rows) / 2 |
| `reconcile … [--output <path>]` | execute through two catalogues (with and without the index) and compare | 0 / **3** (plans disagreed) / 2 |

`--table` is repeatable: a single-table query keeps the bare `--table <jsonl>` form, a join needs
`--table <name>=<jsonl>` for each of the two inputs. `--index` is repeatable and takes a column
(`region`) or, for joins, a qualified `table.column`; an unqualified index column that both inputs
share is a `validation_error`. `--table -` reads rows from stdin. Missing, duplicated or malformed
table bindings are `validation_error`s.

## Joins

One `INNER JOIN` per query, on a single equality between a column of the left input and a column of
the right input. Anything else — a non-equality `ON`, `ON` keys from the same side, a second
`JOIN`, a self join — is a `plan_error`; a syntactically broken `ON` is a `parse_error` with `line`
and `column`.

* **pushdown and pruning per side** — `WHERE` conjuncts that mention only one input move to that
  input's scan (and can be answered by an index there); conjuncts mentioning both sides stay as a
  filter above the join. Each scan reads only the columns the query needs from that table.
* **cost-based join choice** — from the *filtered* row estimates and the catalogue's distinct
  counts the planner costs a hash join in both build directions
  (`HASH_BUILD_COST_PER_ROW = 1.0`, `HASH_PROBE_COST_PER_ROW = 0.5`, so the smaller side builds)
  and, when a join key is indexed, an index nested loop (`INDEX_COST` per outer row). The cheapest
  plan wins; on an exact tie the hash join is preferred, then the build side follows `FROM` order.
  Every candidate and its cost is listed in `notes`, and the plan JSON carries a `join` summary
  (operator, keys, estimated rows, join order) so the choice is observable.
* **deterministic results** — duplicate join keys produce every combination; a null key on either
  side never matches. Without `ORDER BY`, rows come out in the left input's original row order and,
  within one left row, the right input's original order — so hash join (either build direction) and
  index nested loop return byte-identical `columns` and `rows`.
* **`SELECT *` over a join** expands to the left input's columns then the right's; a name both
  inputs share is emitted under its qualified alias (`orders.cust`) so the output record cannot
  collide.

`reconcile` reports `joinOperator` and `joinOrder` for both catalogues, so an index flipping the
plan from `hash-join` to `index-nested-loop` shows up in the report; the results must still be
identical, and a disagreement keeps exit code **3** with both result sets attached.

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
* **cost-based join** — hash join (build side chosen by cost) or index nested loop, decided, not assumed
* **explicit extension surface** — `describe` states what this build does *not* plan (multi-way or
  self joins, outer joins, index-only scans, block indexes), so the next task has a declared boundary
  instead of a guess

`reconcile` is the guarantee that matters: the same statement executed through different physical plans
must produce identical columns and rows. It reports `identical`, both plans' `accessPath`, their
estimated costs, and — when they disagree — both result sets, and exits **3**.

## Guarantees

* Unspecified `--output` writes to stdout; otherwise the result is written to a temporary file and
  atomically replaced, so a failure never corrupts the previous file or leaves a partial one.
* An output path equal to any input table path is rejected as `output_error` **before** anything is read.
* Every error document is `{"error":"<kind>","message":"…", …}`; parse errors carry `line` and `column`,
  row-loading errors carry `line`.

## Layout

```
sqlplan/lexer.py     tokeniser with 1-based positions, comments and string escapes
sqlplan/parser.py    recursive-descent parser and the AST (with to_document); INNER JOIN ... ON
sqlplan/planner.py   logical nodes, rewrites, selectivity and cost model, join choice, catalogue, explain
sqlplan/executor.py  physical execution: scans, filter, hash join, index nested loop, aggregate, project, sort, limit
sqlplan/cli.py       six subcommands and the exit-code contract
tests/               parser positions, plan choices, aggregation semantics, join planning/execution, CLI surface
```
