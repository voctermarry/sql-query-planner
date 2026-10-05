"""Command line entry point.

Subcommands (stdout is JSON only; errors are one JSON document on stderr):

  describe   -- capabilities and exit codes
  parse      -- the AST of one statement
  plan       -- logical and physical plan with estimates and the rewrites that were applied
  explain    -- the same plan plus an indented operator tree for humans
  run        -- execute the statement against one or two JSONL tables
  reconcile  -- execute the statement through two different physical plans and compare the results

Table bindings are repeatable: ``--table orders=orders.jsonl``; a bare ``--table orders.jsonl`` keeps
the single-table form working. Indexes follow the same split: ``--index orders.cid`` for a join, a
bare ``--index region`` for one table. Exit codes are part of the contract: 0 success, 2 input/usage
error, 3 a report was produced whose verdict is negative (no rows, or two plans disagreed). `run` and
`reconcile` write output atomically, and an output path that collides with any input is rejected
before anything is read.
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
    HashJoin,
    IndexNestedLoopJoin,
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


def _table_bindings(statement: Any, values: Sequence[str]) -> list[tuple[str, str]]:
    """Resolve repeatable --table values to (name, path) pairs in FROM order.

    A one-table query keeps the baseline contract verbatim: the single value is a path and is never
    split (so a path that happens to contain '=' still opens). Only a join switches to the
    repeatable ``name=path`` form, in which every binding must be well-formed.
    """
    required = [statement.table, *(join.table for join in statement.joins)]
    if len(required) == 1:
        if len(values) != 1:
            raise ValidationError("a one-table query takes exactly one --table path")
        return [(required[0], values[0])]

    bindings: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValidationError(f"malformed table binding: {value!r} (need 'name=path')", value=value)
        name, path = value.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValidationError(f"malformed table binding: {value!r} (need 'name=path')", value=value)
        if name in bindings:
            raise ValidationError(f"duplicate table binding: {name}", table=name)
        if name not in required:
            raise ValidationError(f"table binding is not used by the query: {name}", table=name, known=required)
        bindings[name] = path
    missing = [name for name in required if name not in bindings]
    if missing:
        raise ValidationError("missing table binding", tables=missing)
    return [(name, bindings[name]) for name in required]


def _index_specs(statement: Any, raw: Sequence[str], table_names: Sequence[str]) -> dict[str, list[str]]:
    specs: dict[str, list[str]] = {name: [] for name in table_names}
    for value in raw:
        if "." in value:
            table, column = value.split(".", 1)
            if not table or not column:
                raise ValidationError(f"malformed index declaration: {value!r} (need 'table.column')", value=value)
            if table not in specs:
                raise ValidationError(f"index on unknown table: {table}", table=table, known=list(table_names))
        else:
            if len(table_names) != 1:
                raise ValidationError("join indexes must be qualified as 'table.column'", value=value)
            table, column = table_names[0], value
        if column not in specs[table]:
            specs[table].append(column)
    return specs


def _catalog_and_tables(
    bindings: Sequence[tuple[str, str]], indexes: dict[str, list[str]]
) -> tuple[Catalog, dict[str, Table]]:
    catalog = Catalog()
    tables: dict[str, Table] = {}
    for name, path in bindings:
        rows = _load_rows(path)
        columns: list[str] = []
        for row in rows:
            for column in row:
                if column not in columns:
                    columns.append(column)
        distinct = {column: len({row.get(column) for row in rows}) for column in columns}
        catalog.add(
            TableInfo(
                name=name,
                columns=tuple(sorted(columns)),
                rows=len(rows),
                distinct=distinct,
                indexes=tuple(indexes.get(name, [])),
            )
        )
        tables[name] = Table(name=name, columns=tuple(sorted(columns)), rows=[dict(row) for row in rows])
    return catalog, tables


def _prepare(args: argparse.Namespace, *, with_index: bool = True) -> tuple[Any, Catalog, dict[str, Table], list[str]]:
    statement = parse(args.sql)
    bindings = _table_bindings(statement, args.table or [])
    input_paths = [path for _, path in bindings]
    _assert_distinct(input_paths, getattr(args, "output", None))
    raw_indexes = args.index or []
    table_names = [name for name, _ in bindings]
    indexes = _index_specs(statement, raw_indexes if with_index else [], table_names) if with_index else {}
    catalog, tables = _catalog_and_tables(bindings, indexes)
    return statement, catalog, tables, input_paths


# -- commands ------------------------------------------------------------------------------------
def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "sql-query-planner",
            "version": __version__,
            "subcommands": ["describe", "explain", "parse", "plan", "reconcile", "run"],
            "aggregates": ["avg", "count", "max", "min", "sum"],
            "operators": [
                "filter",
                "full-scan",
                "hash-join",
                "index-nested-loop-join",
                "index-scan",
                "limit",
                "project",
                "sort",
                "aggregate",
            ],
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
    _, catalog, tables, _ = _prepare(args)
    result = plan(parse(args.sql), catalog)
    _emit(result.to_document())
    return EXIT_OK


def _command_explain(args: argparse.Namespace) -> int:
    _, catalog, _, _ = _prepare(args)
    result = plan(parse(args.sql), catalog)
    document = explain(result)
    document["tree"] = _tree(result.physical)
    _emit(document)
    return EXIT_OK


def _command_run(args: argparse.Namespace) -> int:
    statement, catalog, tables, _ = _prepare(args)
    result = plan(statement, catalog)
    columns, rows = execute(result.physical, tables)
    _write(args.output, [canonical({"columns": columns, "rows": rows, "plan": describe_node(result.physical)["operator"]})])
    return EXIT_OK if rows else EXIT_NEGATIVE


def _join_summary(node: object) -> tuple[str | None, list[str] | None]:
    """The join operator a plan uses and its join order, or (None, None) for a single table."""
    found = _first_join(node)
    if found is None:
        return None, None
    document = describe_node(found)
    return str(document["operator"]), list(document.get("joinOrder", []))


def _first_join(node: object) -> object:
    if isinstance(node, (HashJoin, IndexNestedLoopJoin)):
        return node
    for attribute in ("input", "left", "right", "outer"):
        child = getattr(node, attribute, None)
        if child is not None and hasattr(child, "__dataclass_fields__"):
            found = _first_join(child)
            if found is not None:
                return found
    return None


def _scan_operator(node: object) -> str:
    """The access path actually chosen: the first leaf scan walking the plan.

    Reporting only the root operator made `reconcile` useless: every plan ends in `project`, so two
    plans with different access paths looked identical in the report (the test caught exactly that).
    A join descends through its left/outer input to that input's scan; the join itself is reported
    separately as `joinOperator`.
    """
    for attribute in ("input", "left", "outer", "right"):
        child = getattr(node, attribute, None)
        if child is not None and hasattr(child, "__dataclass_fields__"):
            return _scan_operator(child)
    return str(describe_node(node)["operator"])


def _plan_entry(label: str, chosen: object) -> dict[str, Any]:
    join_operator, join_order = _join_summary(chosen.physical)
    return {
        "catalog": label,
        "estimatedCost": round(chosen.estimated_cost, 3),
        "root": describe_node(chosen.physical)["operator"],
        "accessPath": _scan_operator(chosen.physical),
        "joinOperator": join_operator,
        "joinOrder": join_order,
    }


def _command_reconcile(args: argparse.Namespace) -> int:
    statement = parse(args.sql)
    bindings = _table_bindings(statement, args.table or [])
    _assert_distinct([path for _, path in bindings], args.output)
    table_names = [name for name, _ in bindings]
    indexed_specs = _index_specs(statement, args.index or [], table_names)

    indexed_catalog, tables = _catalog_and_tables(bindings, indexed_specs)
    plain_catalog, _ = _catalog_and_tables(bindings, {})

    chosen = plan(statement, indexed_catalog)
    baseline = plan(statement, plain_catalog)
    first = execute(chosen.physical, tables)
    second = execute(baseline.physical, tables)
    identical = first == second
    report = {
        "sql": args.sql,
        "rows": len(first[1]),
        "identical": identical,
        "plans": [_plan_entry("with-index", chosen), _plan_entry("without-index", baseline)],
    }
    if not identical:
        report["first"] = {"columns": first[0], "rows": first[1]}
        report["second"] = {"columns": second[0], "rows": second[1]}
    _write(args.output, [canonical(report)])
    return EXIT_OK if identical else EXIT_NEGATIVE


def _tree_children(document: dict[str, Any]) -> list[dict[str, Any]]:
    # Binary joins expose left/right; every other node is unary through input. outer/inner duplicate
    # left/right on an index-nested-loop join, so they are deliberately not walked again.
    if isinstance(document.get("left"), dict) and isinstance(document.get("right"), dict):
        return [document["left"], document["right"]]
    child = document.get("input")
    return [child] if isinstance(child, dict) else []


def _node_line(document: dict[str, Any], depth: int) -> str:
    operator = str(document["operator"])
    detail = {
        key: value
        for key, value in document.items()
        if key not in ("operator", "input", "left", "right", "outer", "inner")
    }
    return f"{'  ' * depth}{operator} {canonical(detail) if detail else ''}".rstrip()


def _tree(node: object, depth: int = 0) -> list[str]:
    document = describe_node(node)
    lines = [_node_line(document, depth)]
    for child in _tree_children(document):
        lines.extend(_tree_from_document(child, depth + 1))
    return lines


def _tree_from_document(document: dict[str, Any], depth: int) -> list[str]:
    lines = [_node_line(document, depth)]
    for child in _tree_children(document):
        lines.extend(_tree_from_document(child, depth + 1))
    return lines


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sql", required=True)
    parser.add_argument(
        "--table",
        action="append",
        default=[],
        required=True,
        help="JSONL rows as 'name=path' (repeatable); a bare path works for a one-table query; - is stdin",
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="index on 'table.column' (repeatable); a bare column works for a one-table query",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sql-query-planner", description="SQL parsing, rewriting and cost-based planning")
    parser.add_argument("--version", action="version", version=f"sql-query-planner {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe", help="print capabilities as JSON").set_defaults(handler=_command_describe)

    parse_command = subparsers.add_parser("parse", help="print the AST")
    parse_command.add_argument("--sql", required=True)
    parse_command.set_defaults(handler=_command_parse)

    for name, handler in (("plan", _command_plan), ("explain", _command_explain)):
        command = subparsers.add_parser(name, help="print the plan" if name == "plan" else "print the plan and a tree")
        _add_query_arguments(command)
        command.set_defaults(handler=handler)

    run = subparsers.add_parser("run", help="execute the statement")
    _add_query_arguments(run)
    run.add_argument("--output")
    run.set_defaults(handler=_command_run)

    reconcile = subparsers.add_parser("reconcile", help="execute through two plans and compare")
    _add_query_arguments(reconcile)
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
