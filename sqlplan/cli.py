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


def _catalog_for(name: str, rows: list[dict[str, Any]], indexes: Sequence[str]) -> Catalog:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    distinct: dict[str, int] = {}
    for column in columns:
        distinct[column] = len({row.get(column) for row in rows})
    catalog = Catalog()
    catalog.add(TableInfo(name=name, columns=tuple(sorted(columns)), rows=len(rows), distinct=distinct, indexes=tuple(indexes)))
    return catalog


def _table_for(name: str, rows: list[dict[str, Any]]) -> Table:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return Table(name=name, columns=tuple(sorted(columns)), rows=[dict(row) for row in rows])


def _plan_for(args: argparse.Namespace) -> tuple[Any, Catalog, Table]:
    statement = parse(args.sql)
    rows = _load_rows(args.table)
    catalog = _catalog_for(statement.table, rows, args.index or [])
    return plan(statement, catalog), catalog, _table_for(statement.table, rows)


def _tree(node: object, depth: int = 0) -> list[str]:
    document = describe_node(node)
    operator = str(document["operator"])
    detail = {key: value for key, value in document.items() if key not in ("operator", "input")}
    line = f"{'  ' * depth}{operator} {canonical(detail) if detail else ''}".rstrip()
    lines = [line]
    child = document.get("input")
    if isinstance(child, dict):
        lines.extend(_tree_from_document(child, depth + 1))
    return lines


def _tree_from_document(document: dict[str, Any], depth: int) -> list[str]:
    operator = str(document["operator"])
    detail = {key: value for key, value in document.items() if key not in ("operator", "input")}
    lines = [f"{'  ' * depth}{operator} {canonical(detail) if detail else ''}".rstrip()]
    child = document.get("input")
    if isinstance(child, dict):
        lines.extend(_tree_from_document(child, depth + 1))
    return lines


# -- commands ------------------------------------------------------------------------------------
def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "sql-query-planner",
            "version": __version__,
            "subcommands": ["describe", "explain", "parse", "plan", "reconcile", "run"],
            "aggregates": ["avg", "count", "max", "min", "sum"],
            "operators": ["filter", "full-scan", "index-scan", "limit", "project", "sort", "aggregate"],
            "statistics": "from the table itself: row count and per-column distinct counts; --index declares an index",
            "exitCodes": {"ok": EXIT_OK, "error": EXIT_ERROR, "negativeVerdict": EXIT_NEGATIVE},
            "extensionSurface": list(EXTENSION_SURFACE),
        }
    )
    return EXIT_OK


def _command_parse(args: argparse.Namespace) -> int:
    _emit(parse(args.sql).to_document())
    return EXIT_OK


def _command_plan(args: argparse.Namespace) -> int:
    result, _, _ = _plan_for(args)
    _emit(result.to_document())
    return EXIT_OK


def _command_explain(args: argparse.Namespace) -> int:
    result, _, _ = _plan_for(args)
    document = explain(result)
    document["tree"] = _tree(result.physical)
    _emit(document)
    return EXIT_OK


def _command_run(args: argparse.Namespace) -> int:
    _assert_distinct([args.table], args.output)
    result, _, table = _plan_for(args)
    columns, rows = execute(result.physical, table)
    _write(args.output, [canonical({"columns": columns, "rows": rows, "plan": describe_node(result.physical)["operator"]})])
    return EXIT_OK if rows else EXIT_NEGATIVE


def _scan_operator(node: object) -> str:
    """The access path actually chosen, found by walking to the bottom of the tree.

    Reporting only the root operator made `reconcile` useless: every plan ends in `project`, so two
    plans with different access paths looked identical in the report (the test caught exactly that).
    """
    document = describe_node(node)
    child = document.get("input")
    while isinstance(child, dict):
        document = child
        child = document.get("input")
    return str(document["operator"])


def _command_reconcile(args: argparse.Namespace) -> int:
    _assert_distinct([args.table], args.output)
    statement = parse(args.sql)
    all_rows = _load_rows(args.table)
    indexed = _catalog_for(statement.table, all_rows, args.index or [])
    plain = _catalog_for(statement.table, all_rows, [])
    table = _table_for(statement.table, all_rows)

    chosen = plan(statement, indexed)
    baseline = plan(statement, plain)
    first = execute(chosen.physical, table)
    second = execute(baseline.physical, table)
    identical = first == second
    report = {
        "sql": args.sql,
        "rows": len(first[1]),
        "identical": identical,
        "plans": [
            {
                "catalog": "with-index",
                "estimatedCost": round(chosen.estimated_cost, 3),
                "root": describe_node(chosen.physical)["operator"],
                "accessPath": _scan_operator(chosen.physical),
            },
            {
                "catalog": "without-index",
                "estimatedCost": round(baseline.estimated_cost, 3),
                "root": describe_node(baseline.physical)["operator"],
                "accessPath": _scan_operator(baseline.physical),
            },
        ],
    }
    if not identical:
        report["first"] = {"columns": first[0], "rows": first[1]}
        report["second"] = {"columns": second[0], "rows": second[1]}
    _write(args.output, [canonical(report)])
    return EXIT_OK if identical else EXIT_NEGATIVE


def _add_query_arguments(parser: argparse.ArgumentParser, *, need_table: bool) -> None:
    parser.add_argument("--sql", required=True)
    if need_table:
        parser.add_argument("--table", required=True, help="JSONL rows, or - for stdin")
        parser.add_argument("--index", action="append", default=[], help="column to declare an index on (repeatable)")


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
