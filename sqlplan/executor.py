"""Execution of physical plans over in-memory tables.

The executor is deliberately separate from the planner: the same logical query can be executed through
different physical plans, and the reconciliation command exists to prove they agree. Index scans build
their index on first use and cache it on the table, so `index-scan` is a real access path rather than a
label. Joins come in two physical flavours (hash join, index nested loop); both emit rows in the same
canonical order -- left input in its original row order, and within one left row the right input in its
original row order -- so the algorithm choice never changes the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .errors import PlanError, ValidationError
from .parser import Aggregate, BoolOp, Column, Comparison, InList, IsNull, Literal, Not, Projection, Star
from .planner import (
    FullScan,
    HashJoin,
    IndexNestedLoop,
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


def evaluate(node: object, row: dict[str, object]) -> object:
    """Evaluate a scalar or predicate expression against one row."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Column):
        if node.table:
            qualified = f"{node.table}.{node.name}"
            if qualified in row:
                return row[qualified]
        if node.name in row:
            return row[node.name]
        raise PlanError(f"column not present in row: {node.name}", known=sorted(row))
    if isinstance(node, Comparison):
        left, right = evaluate(node.left, row), evaluate(node.right, row)
        if left is None or right is None:
            return None
        return {
            "=": lambda: left == right,
            "<>": lambda: left != right,
            "<": lambda: left < right,  # type: ignore[operator]
            "<=": lambda: left <= right,  # type: ignore[operator]
            ">": lambda: left > right,  # type: ignore[operator]
            ">=": lambda: left >= right,  # type: ignore[operator]
        }[node.operator]()
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
        argument = "*" if isinstance(expression.argument, Star) else getattr(expression.argument, "name", "?")
        return _aggregate_label(expression)
    if isinstance(expression, Star):
        return "*"
    return "expression"


def execute(node: object, tables: object) -> tuple[list[str], list[dict[str, object]]]:
    """Run a physical plan. `tables` is one Table (single-table plans) or a name -> Table mapping."""
    if isinstance(tables, Table):
        tables = {tables.name: tables}
    return _execute(node, tables)  # type: ignore[arg-type]


def _table_for_node(tables: Mapping[str, Table], name: str) -> Table:
    if name not in tables:
        raise PlanError(f"no rows loaded for table: {name}", known=sorted(tables))
    return tables[name]


def _execute(node: object, tables: Mapping[str, Table]) -> tuple[list[str], list[dict[str, object]]]:
    if isinstance(node, FullScan):
        table = _table_for_node(tables, node.table)
        names = [f"{node.table}.{column}" for column in node.columns] if node.qualify else list(node.columns)
        rows = [{name: row.get(column) for name, column in zip(names, node.columns)} for row in table.rows]
        return names, rows
    if isinstance(node, IndexScan):
        table = _table_for_node(tables, node.table)
        positions = table.index_for(node.column).get(node.value, [])
        names = [f"{node.table}.{column}" for column in node.columns] if node.qualify else list(node.columns)
        rows = [{name: table.rows[position].get(column) for name, column in zip(names, node.columns)} for position in positions]
        return names, rows
    if isinstance(node, HashJoin):
        _, left_rows = _execute(node.left, tables)
        _, right_rows = _execute(node.right, tables)
        build_rows, build_key = (left_rows, node.left_key) if node.build == "left" else (right_rows, node.right_key)
        probe_rows, probe_key = (right_rows, node.right_key) if node.build == "left" else (left_rows, node.left_key)
        hashed: dict[object, list[int]] = {}
        for position, row in enumerate(build_rows):
            key = row.get(build_key)
            if key is None:
                continue  # a null join key never matches
            hashed.setdefault(key, []).append(position)
        pairs: list[tuple[int, int]] = []
        for probe_position, row in enumerate(probe_rows):
            key = row.get(probe_key)
            if key is None:
                continue
            for build_position in hashed.get(key, ()):
                pairs.append((build_position, probe_position) if node.build == "left" else (probe_position, build_position))
        pairs.sort()  # canonical order: left row order, then right row order within one left row
        merged = [{**left_rows[left], **right_rows[right]} for left, right in pairs]
        return list(node.columns), merged
    if isinstance(node, IndexNestedLoop):
        _, outer_rows = _execute(node.outer, tables)
        inner_table = _table_for_node(tables, node.inner_table)
        index = inner_table.index_for(node.inner_key)
        pairs = []
        for outer_position, outer_row in enumerate(outer_rows):
            key = outer_row.get(node.outer_key)
            if key is None:
                continue  # a null join key never matches
            for inner_position in index.get(key, []):
                raw = inner_table.rows[inner_position]
                inner_row = {f"{node.inner_table}.{column}": raw.get(column) for column in node.inner_columns}
                if node.inner_predicate is not None and not _truthy(evaluate(node.inner_predicate, inner_row)):
                    continue
                left, right = (outer_position, inner_position) if node.outer_is_left else (inner_position, outer_position)
                pairs.append((left, right, outer_row, inner_row))
        pairs.sort(key=lambda pair: (pair[0], pair[1]))
        merged = [
            ({**outer, **inner} if node.outer_is_left else {**inner, **outer})
            for _, _, outer, inner in pairs
        ]
        return list(node.columns), merged
    if isinstance(node, PhysicalFilter):
        columns, rows = _execute(node.input, tables)
        return columns, [row for row in rows if _truthy(evaluate(node.predicate, row))]
    if isinstance(node, PhysicalAggregate):
        columns, rows = _execute(node.input, tables)
        names = list(node.group_by) + [_aggregate_label(function) for function in node.aggregates]
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault(tuple(row.get(name) for name in node.group_by), []).append(row)
        if not node.group_by and not groups:
            groups[()] = []
        output: list[dict[str, object]] = []
        for key in sorted(groups, key=lambda item: tuple((value is None, str(value)) for value in item)):
            members = groups[key]
            record: dict[str, object] = {name: value for name, value in zip(node.group_by, key)}
            for function in node.aggregates:
                values = [_aggregate_input(function, member) for member in members]
                record[_aggregate_label(function)] = aggregate_value(function.function, values, function.distinct)
            output.append(record)
        return names, output
    if isinstance(node, PhysicalProject):
        columns, rows = _execute(node.input, tables)
        aggregates = [projection for projection in node.projections if isinstance(projection.expression, Aggregate)]
        if aggregates:
            # aggregates were computed by PhysicalAggregate, which names them "<fn>(<arg>)"
            output = []
            for row in rows:
                record = {}
                for projection in node.projections:
                    expression = projection.expression
                    if isinstance(expression, Star):
                        record.update(row)
                    elif isinstance(expression, Aggregate):
                        record[_projection_label(projection)] = row.get(_aggregate_label(expression))
                    elif isinstance(expression, Column):
                        record[_projection_label(projection)] = _column_lookup(expression, row)
                    else:
                        record[_projection_label(projection)] = evaluate(expression, row)
                output.append(record)
            return [_projection_label(projection) for projection in node.projections], output
        output = []
        for row in rows:
            record = {}
            for projection in node.projections:
                expression = projection.expression
                if isinstance(expression, Star):
                    record.update(row)
                else:
                    record[_projection_label(projection)] = evaluate(expression, row)
            output.append(record)
        return [_projection_label(projection) for projection in node.projections], output
    if isinstance(node, PhysicalSort):
        columns, rows = _execute(node.input, tables)

        def sort_key(row: dict[str, object]) -> tuple:
            key: list[object] = []
            for column, descending in node.keys:
                value = row.get(column)
                if descending:
                    key.append(_Descending(value))
                else:
                    key.append(_Ascending(value))
            return tuple(key)

        return columns, sorted(rows, key=sort_key)
    if isinstance(node, PhysicalLimit):
        columns, rows = _execute(node.input, tables)
        return columns, rows[: node.count]
    raise ValidationError(f"cannot execute node: {type(node).__name__}")


def _column_lookup(column: Column, row: dict[str, object]) -> object:
    """Tolerant column read: qualified key first, bare name as fallback, None when absent."""
    if column.table:
        qualified = f"{column.table}.{column.name}"
        if qualified in row:
            return row[qualified]
    return row.get(column.name)


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
    if isinstance(function.argument, Column):
        return _column_lookup(function.argument, row)
    return row.get(getattr(function.argument, "name", ""))


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
