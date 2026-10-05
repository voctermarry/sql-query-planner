"""Command line entry point.

Subcommands (stdout is JSON only; errors are one JSON document on stderr):

  describe   -- capabilities and exit codes
  parse      -- the AST of one statement
  plan       -- logical and physical plan with estimates and the rewrites that were applied
  explain    -- the same plan plus an indented operator tree for humans
  run        -- execute the statement against a JSONL table
  reconcile  -- execute the statement through two different physical plans and compare the results

Exit codes are part of the contract: 0 success, 2 input/usage error, 3 a report was produced whose
verdict is negative (no rows, or two plans disagreed). `run` and `reconcile` write output atomically,
and an output path that collides with an input is rejected before anything is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Sequence

from . import __version__
from .errors import OutputError, SQLPlanError, ValidationError
from .executor import Table, execute
from .parser import parse
from .planner import (
    EXTENSION_SURFACE,
    Catalog,
    FullScan,
    IndexScan,
    TableInfo,
    describe_node,
    explain,
    join_summary,
    plan,
)

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NEGATIVE = 3


def canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.write(canonical(document) + "\n")


def _read_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError as error:
        raise ValidationError(f"cannot read input: {error.strerror or error}", value=path) from error


def _write(path: str | None, lines: Sequence[str]) -> None:
    payload = "".join(f"{line}\n" for line in lines)
    if not path or path == "-":
        sys.stdout.write(payload)
        return
    directory = os.path.dirname(os.path.abspath(path)) or "."
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False, newline="\n") as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, path)
    except OSError as error:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise OutputError(f"cannot write output: {error.strerror or error}", value=path) from error


def _assert_distinct(inputs: Sequence[str | None], output: str | None) -> None:
    if not output or output == "-":
        return
    target = os.path.abspath(output)
    for source in inputs:
        if source and source != "-" and os.path.abspath(source) == target:
            raise OutputError("output path collides with an input path", value=output)


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(_read_lines(path), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValidationError(f"row {number}: invalid JSON: {error.msg}", line=number) from error
        if not isinstance(document, dict):
            raise ValidationError(f"row {number}: each row must be a JSON object", line=number)
        rows.append(document)
    return rows


def _columns_of(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return columns


def _parse_bindings(entries: Sequence[str], statement: Any) -> dict[str, str]:
    """Map each table the query names to its JSONL path.

    Single-table queries keep the historical bare `--table path`; a join needs `--table
    name=path` for every input. Missing, duplicated or malformed bindings are validation errors.
    """
    needed = [statement.table] + [join.table for join in statement.joins]
    named: dict[str, str] = {}
    anonymous: list[str] = []
    for entry in entries:
        if "=" in entry:
            name, _, path = entry.partition("=")
            if not name or not path:
                raise ValidationError("malformed --table binding, expected name=path", value=entry)
            if name in named:
                raise ValidationError("duplicate table binding", value=name)
            named[name] = path
        else:
            anonymous.append(entry)
    if anonymous:
        if len(needed) > 1:
            raise ValidationError("join queries need --table name=path for every table")
        if named or len(anonymous) > 1:
            raise ValidationError("duplicate table binding", value=statement.table)
        named[statement.table] = anonymous[0]
    missing = [name for name in dict.fromkeys(needed) if name not in named]
    if missing:
        raise ValidationError("missing table binding", value=", ".join(missing))
    extra = [name for name in named if name not in needed]
    if extra:
        raise ValidationError("binding for a table the query does not use", value=", ".join(sorted(extra)))
    return {name: named[name] for name in dict.fromkeys(needed)}


def _resolve_indexes(entries: Sequence[str], columns_by_name: dict[str, list[str]]) -> dict[str, list[str]]:
    """`--index table.column` declares an index; the bare `--index column` form still works when
    exactly one bound table has that column."""
    per_table: dict[str, list[str]] = {name: [] for name in columns_by_name}
    for entry in entries:
        if "." in entry:
            table, _, column = entry.partition(".")
            if table not in columns_by_name:
                raise ValidationError("index declared for an unknown table", value=entry)
        else:
            column = entry
            matches = [name for name, columns in columns_by_name.items() if column in columns]
            if len(matches) > 1:
                raise ValidationError("ambiguous index column, qualify it as table.column", value=entry)
            if not matches:
                raise ValidationError("index on unknown column", value=entry)
            table = matches[0]
        per_table[table].append(column)
    return per_table


def _catalog_for(rows_by_name: dict[str, list[dict[str, Any]]], columns_by_name: dict[str, list[str]], indexes: dict[str, list[str]]) -> Catalog:
    catalog = Catalog()
    for name, rows in rows_by_name.items():
        columns = columns_by_name[name]
        distinct = {column: len({row.get(column) for row in rows}) for column in columns}
        catalog.add(TableInfo(name=name, columns=tuple(sorted(columns)), rows=len(rows), distinct=distinct, indexes=tuple(indexes.get(name, ()))))
    return catalog


def _table_for(name: str, rows: list[dict[str, Any]]) -> Table:
    return Table(name=name, columns=tuple(sorted(_columns_of(rows))), rows=[dict(row) for row in rows])


def _load_query(args: argparse.Namespace) -> tuple[Any, Catalog, dict[str, Table]]:
    """Parse, bind, and load everything a query command needs (statement, catalog, tables)."""
    statement, rows_by_name, columns_by_name = _load_inputs(args)
    indexes = _resolve_indexes(args.index or [], columns_by_name)
    catalog = _catalog_for(rows_by_name, columns_by_name, indexes)
    tables = {name: _table_for(name, rows) for name, rows in rows_by_name.items()}
    return statement, catalog, tables


def _load_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    statement = parse(args.sql)
    bindings = _parse_bindings(args.table, statement)
    rows_by_name = {name: _load_rows(path) for name, path in bindings.items()}
    columns_by_name = {name: _columns_of(rows) for name, rows in rows_by_name.items()}
    return statement, rows_by_name, columns_by_name


def _input_paths(args: argparse.Namespace) -> list[str]:
    """Every input path the command will read (for the output-collision check, before any read)."""
    statement = parse(args.sql)
    return list(_parse_bindings(args.table, statement).values())


def _tree(node: object, depth: int = 0) -> list[str]:
    return _tree_from_document(describe_node(node), depth)


def _tree_from_document(document: dict[str, Any], depth: int, label: str | None = None) -> list[str]:
    operator = str(document["operator"])
    children = ("input", "left", "right", "outer")
    detail = {key: value for key, value in document.items() if key not in ("operator", *children)}
    head = f"{'  ' * depth}{label + ': ' if label else ''}{operator}"
    lines = [f"{head} {canonical(detail) if detail else ''}".rstrip()]
    for key in children:
        child = document.get(key)
        if isinstance(child, dict):
            lines.extend(_tree_from_document(child, depth + 1, None if key == "input" else key))
    return lines


# -- commands ------------------------------------------------------------------------------------
def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "sql-query-planner",
            "version": __version__,
            "subcommands": ["describe", "explain", "parse", "plan", "reconcile", "run"],
            "aggregates": ["avg", "count", "max", "min", "sum"],
            "operators": ["filter", "full-scan", "index-scan", "limit", "project", "sort", "aggregate", "join", "hash-join", "index-nested-loop"],
            "statistics": "from the tables themselves: row count and per-column distinct counts; --index declares an index",
            "exitCodes": {"ok": EXIT_OK, "error": EXIT_ERROR, "negativeVerdict": EXIT_NEGATIVE},
            "extensionSurface": list(EXTENSION_SURFACE),
        }
    )
    return EXIT_OK


def _command_parse(args: argparse.Namespace) -> int:
    _emit(parse(args.sql).to_document())
    return EXIT_OK


def _command_plan(args: argparse.Namespace) -> int:
    statement, catalog, _ = _load_query(args)
    _emit(plan(statement, catalog).to_document())
    return EXIT_OK


def _command_explain(args: argparse.Namespace) -> int:
    statement, catalog, _ = _load_query(args)
    result = plan(statement, catalog)
    document = explain(result)
    document["tree"] = _tree(result.physical)
    _emit(document)
    return EXIT_OK


def _command_run(args: argparse.Namespace) -> int:
    _assert_distinct(_input_paths(args), args.output)
    statement, catalog, tables = _load_query(args)
    result = plan(statement, catalog)
    columns, rows = execute(result.physical, tables)
    _write(args.output, [canonical({"columns": columns, "rows": rows, "plan": describe_node(result.physical)["operator"]})])
    return EXIT_OK if rows else EXIT_NEGATIVE


def _scan_operator(node: object) -> str:
    """The access path actually chosen, found by walking to the bottom of the tree.

    Reporting only the root operator made `reconcile` useless: every plan ends in `project`, so two
    plans with different access paths looked identical in the report (the test caught exactly that).
    For joins the walk follows the left input, so the left table's scan is what gets reported.
    """
    document = describe_node(node)
    child = document.get("input")
    if not isinstance(child, dict):
        child = document.get("left")
    if not isinstance(child, dict):
        child = document.get("outer")
    while isinstance(child, dict):
        document = child
        child = document.get("input")
        if not isinstance(child, dict):
            child = document.get("left")
        if not isinstance(child, dict):
            child = document.get("outer")
    return str(document["operator"])


def _command_reconcile(args: argparse.Namespace) -> int:
    _assert_distinct(_input_paths(args), args.output)
    statement, rows_by_name, columns_by_name = _load_inputs(args)
    indexes = _resolve_indexes(args.index or [], columns_by_name)
    indexed = _catalog_for(rows_by_name, columns_by_name, indexes)
    plain = _catalog_for(rows_by_name, columns_by_name, {name: [] for name in columns_by_name})
    tables = {name: _table_for(name, rows) for name, rows in rows_by_name.items()}

    chosen = plan(statement, indexed)
    baseline = plan(statement, plain)
    first = execute(chosen.physical, tables)
    second = execute(baseline.physical, tables)
    identical = first == second
    report = {
        "sql": args.sql,
        "rows": len(first[1]),
        "identical": identical,
        "plans": [
            _reconcile_plan_entry("with-index", chosen),
            _reconcile_plan_entry("without-index", baseline),
        ],
    }
    if not identical:
        report["first"] = {"columns": first[0], "rows": first[1]}
        report["second"] = {"columns": second[0], "rows": second[1]}
    _write(args.output, [canonical(report)])
    return EXIT_OK if identical else EXIT_NEGATIVE


def _reconcile_plan_entry(label: str, result: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "catalog": label,
        "estimatedCost": round(result.estimated_cost, 3),
        "root": describe_node(result.physical)["operator"],
        "accessPath": _scan_operator(result.physical),
    }
    summary = join_summary(result.physical)
    if summary is not None:
        entry["joinOperator"] = summary["operator"]
        entry["joinOrder"] = summary["order"]
    return entry


def _add_query_arguments(parser: argparse.ArgumentParser, *, need_table: bool) -> None:
    parser.add_argument("--sql", required=True)
    if need_table:
        parser.add_argument("--table", action="append", required=True, help="JSONL rows as path or name=path (repeatable), or - for stdin")
        parser.add_argument("--index", action="append", default=[], help="column (or table.column) to declare an index on (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sql-query-planner", description="SQL parsing, rewriting and cost-based planning")
    parser.add_argument("--version", action="version", version=f"sql-query-planner {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe", help="print capabilities as JSON").set_defaults(handler=_command_describe)

    parse_command = subparsers.add_parser("parse", help="print the AST")
    parse_command.add_argument("--sql", required=True)
    parse_command.set_defaults(handler=_command_parse)

    for name, handler, needs_table in (("plan", _command_plan, True), ("explain", _command_explain, True)):
        command = subparsers.add_parser(name, help="print the plan" if name == "plan" else "print the plan and a tree")
        _add_query_arguments(command, need_table=needs_table)
        command.set_defaults(handler=handler)

    run = subparsers.add_parser("run", help="execute the statement")
    _add_query_arguments(run, need_table=True)
    run.add_argument("--output")
    run.set_defaults(handler=_command_run)

    reconcile = subparsers.add_parser("reconcile", help="execute through two plans and compare")
    _add_query_arguments(reconcile, need_table=True)
    reconcile.add_argument("--output")
    reconcile.set_defaults(handler=_command_reconcile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except SQLPlanError as error:
        sys.stderr.write(canonical(error.to_document()) + "\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
