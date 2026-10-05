"""Execution of a physical plan over one or two in-memory tables.

The executor is deliberately separate from the planner: the same logical query can be executed through
different physical plans, and the reconciliation command exists to prove they agree. Index scans build
their index on first use and cache it on the table, so `index-scan` is a real access path rather than a
label.

Rows below a join are local to one table and keyed by plain column names. A join emits rows keyed
"table.column"; every operator above a join therefore sees qualified column references. Hash joins in
either build direction and index-nested-loop joins all canonicalise their output to (left original
position, right original position), so the physical algorithm never affects row order or content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .errors import PlanError, ValidationError
from .parser import Aggregate, BoolOp, Column, Comparison, InList, IsNull, Literal, Not, Projection, Star
from .planner import (
    FullScan,
    HashJoin,
    IndexNestedLoopJoin,
    IndexScan,
    PhysicalAggregate,
    PhysicalFilter,
    PhysicalLimit,
    PhysicalProject,
    PhysicalSort,
)


@dataclass(slots=True)
class Table:
    name: str
    columns: tuple[str, ...]
    rows: list[dict[str, object]] = field(default_factory=list)
    _indexes: dict[str, dict[object, list[int]]] = field(default_factory=dict, init=False)

    def index_for(self, column: str) -> dict[object, list[int]]:
        """Built once per column and reused; the planner is told which columns have one."""
        if column not in self.columns:
            raise ValidationError(f"no such column to index: {column}", table=self.name)
        if column not in self._indexes:
            table: dict[object, list[int]] = {}
            for position, row in enumerate(self.rows):
                table.setdefault(row.get(column), []).append(position)
            self._indexes[column] = table
        return self._indexes[column]


def _row_key(column: Column) -> str:
    return f"{column.table}.{column.name}" if column.table is not None else column.name


def evaluate(node: object, row: dict[str, object]) -> object:
    """Evaluate a scalar or predicate expression against one row.

    Below a join rows use plain column names; above a join they use "table.column" and the Column node
    carries that table.
    """
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Column):
        key = _row_key(node)
        if key not in row and node.table is not None and node.name in row:
            key = node.name  # a localized predicate evaluated on local rows
        if key not in row:
            raise PlanError(f"column not present in row: {key}", known=sorted(row))
        return row[key]
    if isinstance(node, Comparison):
        left, right = evaluate(node.left, row), evaluate(node.right, row)
        if left is None or right is None:
            return None
        try:
            return {
                "=": lambda: left == right,
                "<>": lambda: left != right,
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                ">": lambda: left > right,
                ">=": lambda: left >= right,
            }[node.operator]()
        except TypeError:
            return None
    if isinstance(node, InList):
        value = evaluate(node.operand, row)
        if value is None:
            return None
        hit = any(value == evaluate(candidate, row) for candidate in node.values)
        return (not hit) if node.negated else hit
    if isinstance(node, IsNull):
        missing = evaluate(node.operand, row) is None
        return (not missing) if node.negated else missing
    if isinstance(node, Not):
        inner = evaluate(node.operand, row)
        return None if inner is None else (not inner)
    if isinstance(node, BoolOp):
        values = [evaluate(operand, row) for operand in node.operands]
        if node.operator == "and":
            if any(value is False for value in values):
                return False
            return None if any(value is None for value in values) else True
        if any(value is True for value in values):
            return True
        return None if any(value is None for value in values) else False
    raise ValidationError(f"cannot evaluate expression: {type(node).__name__}")


def _truthy(value: object) -> bool:
    return value is True


def aggregate_value(function: str, values: list[object], distinct: bool) -> object:
    if distinct:
        seen: list[object] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        values = seen
    present = [value for value in values if value is not None]
    if function == "count":
        return len(present)
    if not present:
        return None
    numbers = [float(value) for value in present]  # type: ignore[arg-type]
    if function == "sum":
        return sum(numbers)
    if function == "min":
        return min(numbers)
    if function == "max":
        return max(numbers)
    if function == "avg":
        return sum(numbers) / len(numbers)
    raise ValidationError(f"unknown aggregate: {function}")


def _projection_label(projection: Projection) -> str:
    if projection.alias:
        return projection.alias
    expression = projection.expression
    if isinstance(expression, Column):
        return expression.name
    if isinstance(expression, Aggregate):
        return _aggregate_label(expression)
    if isinstance(expression, Star):
        return "*"
    return "expression"


def execute(node: object, tables: "Table | dict[str, Table]") -> tuple[list[str], list[dict[str, object]]]:
    if isinstance(tables, Table):
        tables = {tables.name: tables}
    columns, rows, _ = _execute(node, tables)
    return columns, rows


def _execute(
    node: object, tables: "dict[str, Table]"
) -> tuple[list[str], list[dict[str, object]], list[int] | None]:
    """Internal form: scans also report each row's original position in its table."""
    if isinstance(node, FullScan):
        table = tables[node.table]
        rows = [{column: row.get(column) for column in node.columns} for row in table.rows]
        return list(node.columns), rows, list(range(len(table.rows)))
    if isinstance(node, IndexScan):
        table = tables[node.table]
        positions = list(table.index_for(node.column).get(node.value, []))
        rows = [{column: table.rows[position].get(column) for column in node.columns} for position in positions]
        return list(node.columns), rows, positions
    if isinstance(node, PhysicalFilter):
        columns, rows, positions = _execute(node.input, tables)
        if positions is None:
            return columns, [row for row in rows if _truthy(evaluate(node.predicate, row))], None
        kept = [
            (row, position)
            for row, position in zip(rows, positions)
            if _truthy(evaluate(node.predicate, row))
        ]
        return columns, [row for row, _ in kept], [position for _, position in kept]
    if isinstance(node, (HashJoin, IndexNestedLoopJoin)):
        columns, rows = _execute_join(node, tables)
        return columns, rows, None
    if isinstance(node, PhysicalAggregate):
        columns, rows, _ = _execute(node.input, tables)
        group_keys = list(node.group_by)
        names = list(node.group_labels or group_keys) + [_aggregate_label(function) for function in node.aggregates]
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault(tuple(row.get(name) for name in group_keys), []).append(row)
        if not node.group_by and not groups:
            groups[()] = []
        output: list[dict[str, object]] = []
        for key in sorted(groups, key=lambda item: tuple((value is None, str(value)) for value in item)):
            members = groups[key]
            labels = node.group_labels or group_keys
            record: dict[str, object] = {name: value for name, value in zip(labels, key)}
            for function in node.aggregates:
                values = [_aggregate_input(function, member) for member in members]
                record[_aggregate_label(function)] = aggregate_value(function.function, values, function.distinct)
            output.append(record)
        return names, output, None
    if isinstance(node, PhysicalProject):
        columns, rows, _ = _execute(node.input, tables)
        output: list[dict[str, object]] = []
        for row in rows:
            record: dict[str, object] = {}
            for projection in node.projections:
                expression = projection.expression
                if isinstance(expression, Star):
                    record.update(row)
                elif isinstance(expression, Aggregate):
                    record[_projection_label(projection)] = row.get(_aggregate_label(expression))
                else:
                    record[_projection_label(projection)] = evaluate(expression, row)
            output.append(record)
        return _project_header(node.projections, columns), output, None
    if isinstance(node, PhysicalSort):
        columns, rows, _ = _execute(node.input, tables)

        def sort_key(row: dict[str, object]) -> tuple:
            key: list[object] = []
            for name, table_name, descending in node.keys:
                value = row.get(name)
                if value is None and table_name is not None:
                    value = row.get(f"{table_name}.{name}")
                key.append(_Descending(value) if descending else _Ascending(value))
            return tuple(key)

        return columns, sorted(rows, key=sort_key), None
    if isinstance(node, PhysicalLimit):
        columns, rows, _ = _execute(node.input, tables)
        return columns, rows[: node.count], None
    raise ValidationError(f"cannot execute node: {type(node).__name__}")


def _project_header(projections: tuple[Projection, ...], input_columns: list[str]) -> list[str]:
    """The output column list of a projection, known before any row is seen.

    A star expands to the input's columns at its position in the SELECT list: the scan's columns for
    one table, the join's qualified "table.column" columns above a join. Explicit projections keep
    their labels. Because a JSON object keeps one key per name, the header keeps the first occurrence
    of each name and never contains a literal "*".
    """
    header: list[str] = []
    for projection in projections:
        names = input_columns if isinstance(projection.expression, Star) else [_projection_label(projection)]
        for name in names:
            if name not in header:
                header.append(name)
    return header


def _merged_row(
    left_table: str,
    left_columns: Iterable[str],
    left_row: dict[str, object],
    right_table: str,
    right_columns: Iterable[str],
    right_row: dict[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    for column in left_columns:
        merged[f"{left_table}.{column}"] = left_row.get(column)
    for column in right_columns:
        merged[f"{right_table}.{column}"] = right_row.get(column)
    return merged


def _execute_join(node: object, tables: dict[str, Table]) -> tuple[list[str], list[dict[str, object]]]:
    if isinstance(node, HashJoin):
        left_columns, left_rows, left_positions = _execute(node.left, tables)
        right_columns, right_rows, right_positions = _execute(node.right, tables)
        join_columns = [left_columns, right_columns]
        pairs: list[tuple[int, int, dict[str, object]]] = []
        if node.build_side == "right":
            index = _hash_by_key(right_rows, right_positions, node.right_key)
            for left_row, left_position in zip(left_rows, left_positions or []):
                for right_position, right_row in index.get(left_row.get(node.left_key), []):
                    pairs.append((left_position, right_position, _merged_row(node.left_table, left_columns, left_row, node.right_table, right_columns, right_row)))
        else:
            index = _hash_by_key(left_rows, left_positions, node.left_key)
            for right_row, right_position in zip(right_rows, right_positions or []):
                for left_position, left_row in index.get(right_row.get(node.right_key), []):
                    pairs.append((left_position, right_position, _merged_row(node.left_table, left_columns, left_row, node.right_table, right_columns, right_row)))
    elif isinstance(node, IndexNestedLoopJoin):
        outer_columns, outer_rows, outer_positions = _execute(node.outer, tables)
        inner_table = tables[node.inner_table]
        index = inner_table.index_for(node.inner_index_column)
        if node.outer_side == "left":
            join_columns = [outer_columns, list(node.inner_columns)]
            pairs = _inlj_pairs(node, outer_rows, outer_positions or [], outer_columns, inner_table, index, left_outer=True)
        else:
            join_columns = [list(node.inner_columns), outer_columns]
            pairs = _inlj_pairs(node, outer_rows, outer_positions or [], outer_columns, inner_table, index, left_outer=False)
    else:  # pragma: no cover - guarded by caller
        raise ValidationError(f"cannot execute join node: {type(node).__name__}")

    # One canonical order regardless of algorithm: logical left original order, then right. The
    # column list comes from the inputs' schemas, not from emitted pairs, so an empty join still
    # reports every qualified "table.column" it would have produced.
    pairs.sort(key=lambda item: (item[0], item[1]))
    columns = [f"{node.left_table}.{column}" for column in join_columns[0]] + [
        f"{node.right_table}.{column}" for column in join_columns[1]
    ]
    return columns, [row for _, _, row in pairs]


def _hash_by_key(rows: list[dict[str, object]], positions: list[int], key: str) -> dict[object, list[tuple[int, dict[str, object]]]]:
    table: dict[object, list[tuple[int, dict[str, object]]]] = {}
    for row, position in zip(rows, positions):
        value = row.get(key)
        if value is None:  # a null join key never matches
            continue
        table.setdefault(value, []).append((position, row))
    return table


def _inlj_pairs(
    node: IndexNestedLoopJoin,
    outer_rows: list[dict[str, object]],
    outer_positions: list[int],
    outer_columns: list[str],
    inner_table: Table,
    index: dict[object, list[int]],
    *,
    left_outer: bool,
) -> list[tuple[int, int, dict[str, object]]]:
    outer_key = node.left_key if left_outer else node.right_key
    pairs: list[tuple[int, int, dict[str, object]]] = []
    for outer_row, outer_position in zip(outer_rows, outer_positions):
        value = outer_row.get(outer_key)
        if value is None:
            continue
        for inner_position in index.get(value, []):
            inner_row_full = inner_table.rows[inner_position]
            inner_row = {column: inner_row_full.get(column) for column in node.inner_columns}
            if not all(_truthy(evaluate(predicate, inner_row)) for predicate in node.inner_predicates):
                continue
            if left_outer:
                merged = _merged_row(node.left_table, outer_columns, outer_row, node.right_table, node.inner_columns, inner_row)
                pairs.append((outer_position, inner_position, merged))
            else:
                merged = _merged_row(node.left_table, node.inner_columns, inner_row, node.right_table, outer_columns, outer_row)
                pairs.append((inner_position, outer_position, merged))
    return pairs


def _aggregate_argument(function: Aggregate) -> str:
    if isinstance(function.argument, Star):
        return "*"
    return getattr(function.argument, "name", "?")


def _aggregate_label(function: Aggregate) -> str:
    """One place decides an aggregate's output column name.

    The aggregate stage names its column and the projection stage looks that name up, so the two must
    agree by construction: a first version spelled DISTINCT in one place and not the other, and the
    projection then raised KeyError('count(DISTINCT region)').
    """
    distinct = "DISTINCT " if function.distinct else ""
    return f"{function.function}({distinct}{_aggregate_argument(function)})"


def _aggregate_input(function: Aggregate, row: dict[str, object]) -> object:
    if isinstance(function.argument, Star):
        return 1
    if isinstance(function.argument, Literal):
        return function.argument.value
    return row.get(_row_key(function.argument))


class _Ascending:
    """Sort keys: NULLs last in both directions, values compared as floats when possible."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def _pair(self) -> tuple:
        return (self.value is None, 0 if isinstance(self.value, (int, float)) else 1, "" if self.value is None else str(self.value))

    def __lt__(self, other: "_Ascending") -> bool:
        left, right = self._pair(), other._pair()
        if left[0] != right[0]:
            return left[0] < right[0]
        if left[0]:
            return False
        if isinstance(self.value, (int, float)) and isinstance(other.value, (int, float)):
            return float(self.value) < float(other.value)
        return left[2] < right[2]


class _Descending(_Ascending):
    def __lt__(self, other: "_Ascending") -> bool:
        left, right = self._pair(), other._pair()
        if left[0] != right[0]:
            return left[0] < right[0]
        if left[0]:
            return False
        if isinstance(self.value, (int, float)) and isinstance(other.value, (int, float)):
            return float(self.value) > float(other.value)
        return left[2] > right[2]


def sortable(values: Iterable[object]) -> list[object]:
    return sorted(values, key=lambda value: _Ascending(value))
