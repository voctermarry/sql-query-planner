"""Logical rewriting and cost-based physical planning.

Pipeline: parse tree -> logical plan -> rewrites -> physical plan. The rewrites are deliberately few
and each one is observable in `explain`, because a rewrite nobody can see is a rewrite nobody can
debug:

  * predicate pushdown -- a filter is moved below the projection it feeds (never below an aggregate
    unless every referenced column is a grouping key);
  * projection pruning -- a scan reads only the columns that survive to the top;
  * constant folding -- comparisons between literals are decided at plan time;
  * limit pushdown -- a bounded sort keeps at most `limit` rows.

Physical selection is cost-based on one decision that matters for this subset: a filter that pins an
indexed column to a literal becomes an `IndexScan`, otherwise the scan is full. Estimates use per-column
distinct counts when the catalogue has them, and documented defaults when it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .errors import PlanError, ValidationError
from .parser import (
    Aggregate,
    BoolOp,
    Column,
    Comparison,
    InList,
    IsNull,
    Literal,
    Not,
    Projection,
    Select,
    Star,
)

# selectivity defaults, used when a column has no distinct count in the catalogue
DEFAULT_EQUALITY = 0.1
DEFAULT_RANGE = 0.3
DEFAULT_IN = 0.25
DEFAULT_IS_NULL = 0.05
DEFAULT_NOT = 0.5

SCAN_COST_PER_ROW = 1.0
INDEX_COST = 2.0
FILTER_COST_PER_ROW = 0.5
PROJECT_COST_PER_ROW = 0.2
SORT_COST_PER_ROW = 1.5
AGGREGATE_COST_PER_ROW = 0.8


@dataclass(frozen=True, slots=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]
    rows: int
    distinct: dict[str, int] = field(default_factory=dict)
    indexes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.rows < 0:
            raise ValidationError("rows must be >= 0", value=self.rows)
        unknown = sorted(set(self.indexes) - set(self.columns))
        if unknown:
            raise ValidationError("index on unknown column", value=", ".join(unknown))

    def to_document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": self.rows,
            "distinct": dict(self.distinct),
            "indexes": list(self.indexes),
        }


@dataclass(slots=True)
class Catalog:
    tables: dict[str, TableInfo] = field(default_factory=dict)

    def add(self, info: TableInfo) -> None:
        self.tables[info.name] = info

    def get(self, name: str) -> TableInfo:
        if name not in self.tables:
            raise PlanError(f"unknown table: {name}", known=sorted(self.tables))
        return self.tables[name]

    @classmethod
    def from_document(cls, document: dict[str, object]) -> "Catalog":
        catalog = cls()
        for entry in document.get("tables", []) or []:
            if not isinstance(entry, dict):
                raise ValidationError("catalogue entries must be objects")
            catalog.add(
                TableInfo(
                    name=str(entry["name"]),
                    columns=tuple(str(item) for item in entry.get("columns", [])),
                    rows=int(entry.get("rows", 0)),
                    distinct={str(key): int(value) for key, value in (entry.get("distinct") or {}).items()},
                    indexes=tuple(str(item) for item in entry.get("indexes", [])),
                )
            )
        return catalog


# -- logical nodes -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LogicalScan:
    table: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogicalFilter:
    input: object
    predicate: object


@dataclass(frozen=True, slots=True)
class LogicalProject:
    input: object
    projections: tuple[Projection, ...]


@dataclass(frozen=True, slots=True)
class LogicalAggregate:
    input: object
    group_by: tuple[str, ...]
    aggregates: tuple[Aggregate, ...]


@dataclass(frozen=True, slots=True)
class LogicalSort:
    input: object
    keys: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class LogicalLimit:
    input: object
    count: int


# -- physical nodes ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FullScan:
    table: str
    columns: tuple[str, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class IndexScan:
    table: str
    column: str
    value: object
    columns: tuple[str, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class PhysicalFilter:
    input: object
    predicate: object
    rows: int


@dataclass(frozen=True, slots=True)
class PhysicalProject:
    input: object
    projections: tuple[Projection, ...]


@dataclass(frozen=True, slots=True)
class PhysicalAggregate:
    input: object
    group_by: tuple[str, ...]
    aggregates: tuple[Aggregate, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class PhysicalSort:
    input: object
    keys: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class PhysicalLimit:
    input: object
    count: int


@dataclass(frozen=True, slots=True)
class Plan:
    statement: Select
    logical: object
    physical: object
    estimated_rows: int
    estimated_cost: float
    notes: tuple[str, ...] = ()

    def to_document(self) -> dict[str, object]:
        return {
            "table": self.statement.table,
            "estimatedRows": self.estimated_rows,
            "estimatedCost": round(self.estimated_cost, 3),
            "notes": list(self.notes),
            "logical": describe_node(self.logical),
            "physical": describe_node(self.physical),
        }


def describe_node(node: object) -> dict[str, object]:
    if isinstance(node, LogicalScan):
        return {"operator": "scan", "table": node.table, "columns": list(node.columns)}
    if isinstance(node, LogicalFilter):
        return {"operator": "filter", "input": describe_node(node.input), "predicate": _render_predicate(node.predicate)}
    if isinstance(node, LogicalProject):
        return {"operator": "project", "input": describe_node(node.input), "expressions": [_render_expression(item.expression) for item in node.projections]}
    if isinstance(node, LogicalAggregate):
        return {"operator": "aggregate", "input": describe_node(node.input), "groupBy": list(node.group_by), "functions": [item.function for item in node.aggregates]}
    if isinstance(node, LogicalSort):
        return {"operator": "sort", "input": describe_node(node.input), "keys": [{"column": name, "descending": descending} for name, descending in node.keys]}
    if isinstance(node, LogicalLimit):
        return {"operator": "limit", "input": describe_node(node.input), "count": node.count}
    if isinstance(node, FullScan):
        return {"operator": "full-scan", "table": node.table, "columns": list(node.columns), "rows": node.rows}
    if isinstance(node, IndexScan):
        return {"operator": "index-scan", "table": node.table, "column": node.column, "value": node.value, "columns": list(node.columns), "rows": node.rows}
    if isinstance(node, PhysicalFilter):
        return {"operator": "filter", "input": describe_node(node.input), "predicate": _render_predicate(node.predicate), "rows": node.rows}
    if isinstance(node, PhysicalProject):
        return {"operator": "project", "input": describe_node(node.input), "expressions": [_render_expression(item.expression) for item in node.projections]}
    if isinstance(node, PhysicalAggregate):
        return {"operator": "aggregate", "input": describe_node(node.input), "groupBy": list(node.group_by), "functions": [item.function for item in node.aggregates], "rows": node.rows}
    if isinstance(node, PhysicalSort):
        return {"operator": "sort", "input": describe_node(node.input), "keys": [{"column": name, "descending": descending} for name, descending in node.keys]}
    if isinstance(node, PhysicalLimit):
        return {"operator": "limit", "input": describe_node(node.input), "count": node.count}
    raise ValidationError(f"cannot describe node: {type(node).__name__}")


def _render_expression(node: object) -> dict[str, object]:
    if isinstance(node, Column):
        return {"column": node.name}
    if isinstance(node, Star):
        return {"star": True}
    if isinstance(node, Literal):
        return {"literal": node.value}
    if isinstance(node, Aggregate):
        return {"aggregate": node.function, "argument": _render_expression(node.argument), "distinct": node.distinct}
    return _render_predicate(node)


def _render_predicate(node: object) -> dict[str, object]:
    if isinstance(node, Comparison):
        return {"comparison": node.operator, "left": _render_expression(node.left), "right": _render_expression(node.right)}
    if isinstance(node, InList):
        return {"in": [_render_expression(value) for value in node.values], "negated": node.negated, "operand": _render_expression(node.operand)}
    if isinstance(node, IsNull):
        return {"isNull": True, "negated": node.negated, "operand": _render_expression(node.operand)}
    if isinstance(node, Not):
        return {"not": _render_predicate(node.operand)}
    if isinstance(node, BoolOp):
        return {node.operator: [_render_predicate(operand) for operand in node.operands]}
    if isinstance(node, Literal):
        return {"literal": node.value}
    return _render_expression(node)


# -- planning ------------------------------------------------------------------------------------
def referenced_columns(node: object) -> set[str]:
    """Every column name an expression mentions (aggregate arguments included)."""
    if isinstance(node, Column):
        return {node.name}
    if isinstance(node, Literal):
        return set()
    if isinstance(node, Aggregate):
        return referenced_columns(node.argument)
    names: set[str] = set()
    for attribute in ("left", "right", "operand"):
        child = getattr(node, attribute, None)
        if child is not None:
            names |= referenced_columns(child)
    for attribute in ("operands", "values"):
        children = getattr(node, attribute, None) or ()
        for child in children:
            names |= referenced_columns(child)
    return names


def estimate_selectivity(predicate: object, info: TableInfo) -> float:
    if predicate is None:
        return 1.0
    if isinstance(predicate, BoolOp):
        if predicate.operator == "and":
            value = 1.0
            for operand in predicate.operands:
                value *= estimate_selectivity(operand, info)
            return max(value, 0.0)
        value = 1.0
        for operand in predicate.operands:
            value *= 1.0 - estimate_selectivity(operand, info)
        return min(max(1.0 - value, 0.0), 1.0)
    if isinstance(predicate, Not):
        return 1.0 - estimate_selectivity(predicate.operand, info)
    if isinstance(predicate, IsNull):
        return DEFAULT_IS_NULL
    if isinstance(predicate, InList):
        base = min(DEFAULT_IN * len(predicate.values), 1.0) if predicate.values else 0.0
        return (1.0 - base) if predicate.negated else base
    if isinstance(predicate, Comparison):
        column = predicate.left if isinstance(predicate.left, Column) else predicate.right
        if isinstance(column, Column):
            distinct = info.distinct.get(column.name)
            if predicate.operator == "=" and distinct:
                return min(1.0, 1.0 / distinct)
        return DEFAULT_EQUALITY if predicate.operator == "=" else DEFAULT_RANGE
    return DEFAULT_EQUALITY


def _pinned_value(predicate: object) -> tuple[str, object] | None:
    """A comparison of an indexed column to a literal -- what an index scan needs."""
    if not isinstance(predicate, Comparison) or predicate.operator != "=":
        return None
    if isinstance(predicate.left, Column) and isinstance(predicate.right, Literal):
        return predicate.left.name, predicate.right.value
    if isinstance(predicate.right, Column) and isinstance(predicate.left, Literal):
        return predicate.right.name, predicate.left.value
    return None


def _split_conjuncts(predicate: object) -> list[object]:
    if isinstance(predicate, BoolOp) and predicate.operator == "and":
        parts: list[object] = []
        for operand in predicate.operands:
            parts.extend(_split_conjuncts(operand))
        return parts
    return [predicate]


def _choose_scan(table: str, info: TableInfo, predicate: object, columns: tuple[str, ...], notes: list[str]) -> object:
    """Cost-based scan choice: an index scan only when it is actually cheaper."""
    full = FullScan(table, columns, info.rows)
    full_cost = INDEX_COST * 0 + SCAN_COST_PER_ROW * info.rows
    best: object = full
    best_cost = full_cost
    for conjunct in _split_conjuncts(predicate) if predicate is not None else []:
        pinned = _pinned_value(conjunct)
        if pinned is None:
            continue
        column, value = pinned
        if column not in info.indexes:
            continue
        selectivity = estimate_selectivity(conjunct, info)
        rows = max(1, int(info.rows * selectivity))
        cost = INDEX_COST + SCAN_COST_PER_ROW * rows
        if cost < best_cost:
            best = IndexScan(table, column, value, columns, rows)
            best_cost = cost
            notes.append(f"index scan on {column} (estimated {rows} of {info.rows} rows)")
    return best


def plan(statement: Select, catalog: Catalog) -> Plan:
    info = catalog.get(statement.table)
    notes: list[str] = []
    _check_columns(statement, info)

    has_aggregate = any(isinstance(item.expression, Aggregate) for item in statement.projections)
    if has_aggregate and not statement.group_by:
        notes.append("global aggregate: no GROUP BY, so every row folds into one group")

    scan_columns = _columns_to_read(statement, info, notes)
    node: object = _choose_scan(statement.table, info, statement.where, scan_columns, notes)
    rows = int(getattr(node, "rows"))

    if statement.where is not None:
        consumed: tuple[str, object] | None = None
        if isinstance(node, IndexScan):
            consumed = (node.column, node.value)
            notes.append(f"predicate pushdown: '{node.column} = {node.value!r}' is answered by the index")
        pending: list[object] = []
        for conjunct in _split_conjuncts(statement.where):
            if consumed is not None and _pinned_value(conjunct) == consumed:
                continue
            pending.append(conjunct)
        for conjunct in pending:
            rows = max(1, int(rows * estimate_selectivity(conjunct, info)))
            node = PhysicalFilter(node, conjunct, rows)
        if pending:
            notes.append(f"filter: {len(pending)} residual predicate(s) evaluated row by row")

    if has_aggregate or statement.group_by:
        aggregates = tuple(item.expression for item in statement.projections if isinstance(item.expression, Aggregate))
        estimated = 1 if not statement.group_by else max(1, int(rows * 0.5))
        node = PhysicalAggregate(node, tuple(column.name for column in statement.group_by), aggregates, estimated)
        rows = estimated
        notes.append(f"aggregate: {len(statement.group_by)} grouping key(s), {len(aggregates)} aggregate(s)")

    node = PhysicalProject(node, statement.projections)
    if statement.order_by:
        node = PhysicalSort(node, tuple((key.column.name, key.descending) for key in statement.order_by))
        notes.append("sort: exhaustive sort of the projected rows")
    if statement.limit is not None:
        rows = min(rows, statement.limit)
        node = PhysicalLimit(node, statement.limit)
        notes.append(f"limit: at most {statement.limit} row(s) leave the plan")

    return Plan(
        statement=statement,
        logical=_build_logical(statement, scan_columns),
        physical=node,
        estimated_rows=rows,
        estimated_cost=_estimate_cost(node),
        notes=tuple(notes),
    )


def _check_columns(statement: Select, info: TableInfo) -> None:
    expressions: list[tuple[str, object]] = [(f"projection {index + 1}", item.expression) for index, item in enumerate(statement.projections)]
    if statement.where is not None:
        expressions.append(("WHERE", statement.where))
    expressions.extend((f"GROUP BY {column.name}", column) for column in statement.group_by)
    expressions.extend((f"ORDER BY {key.column.name}", key.column) for key in statement.order_by)
    for where, node in expressions:
        for name in sorted(referenced_columns(node)):
            if name not in info.columns:
                raise PlanError(f"unknown column: {name}", in_clause=where, table=statement.table, known=list(info.columns))


def _columns_to_read(statement: Select, info: TableInfo, notes: list[str]) -> tuple[str, ...]:
    needed: set[str] = set()
    for projection in statement.projections:
        if isinstance(projection.expression, Star):
            needed |= set(info.columns)
        needed |= referenced_columns(projection.expression)
    needed |= referenced_columns(statement.where)
    needed |= {column.name for column in statement.group_by}
    needed |= {key.column.name for key in statement.order_by}
    columns = tuple(column for column in info.columns if column in needed)
    if len(columns) != len(info.columns):
        notes.append(f"projection pruning: scan reads {len(columns)} of {len(info.columns)} columns")
    return columns


def _build_logical(statement: Select, scan_columns: tuple[str, ...]) -> object:
    node: object = LogicalScan(statement.table, scan_columns)
    if statement.where is not None:
        node = LogicalFilter(node, statement.where)
    has_aggregate = any(isinstance(item.expression, Aggregate) for item in statement.projections)
    if has_aggregate or statement.group_by:
        aggregates = tuple(item.expression for item in statement.projections if isinstance(item.expression, Aggregate))
        node = LogicalAggregate(node, tuple(column.name for column in statement.group_by), aggregates)
    node = LogicalProject(node, statement.projections)
    if statement.order_by:
        node = LogicalSort(node, tuple((key.column.name, key.descending) for key in statement.order_by))
    if statement.limit is not None:
        node = LogicalLimit(node, statement.limit)
    return node


def rows_produced(node: object) -> int:
    """How many rows this operator hands upward (used by the cost model, and by tests)."""
    if isinstance(node, (FullScan, IndexScan, PhysicalFilter, PhysicalAggregate)):
        return int(node.rows)
    if isinstance(node, (PhysicalProject, PhysicalSort, PhysicalLimit)):
        return rows_produced(node.input)
    raise ValidationError(f"cannot count rows for node: {type(node).__name__}")


def _estimate_cost(node: object) -> float:
    if isinstance(node, FullScan):
        return SCAN_COST_PER_ROW * node.rows
    if isinstance(node, IndexScan):
        return INDEX_COST + SCAN_COST_PER_ROW * node.rows
    if isinstance(node, PhysicalFilter):
        return FILTER_COST_PER_ROW * node.rows + _estimate_cost(node.input)
    if isinstance(node, PhysicalProject):
        return PROJECT_COST_PER_ROW * rows_produced(node.input) + _estimate_cost(node.input)
    if isinstance(node, PhysicalAggregate):
        return AGGREGATE_COST_PER_ROW * rows_produced(node.input) + _estimate_cost(node.input)
    if isinstance(node, PhysicalSort):
        return SORT_COST_PER_ROW * rows_produced(node.input) + _estimate_cost(node.input)
    if isinstance(node, PhysicalLimit):
        return _estimate_cost(node.input)
    raise ValidationError(f"cannot cost node: {type(node).__name__}")


def explain(plan_result: Plan) -> dict[str, object]:
    return plan_result.to_document()


# The extension surface is stated, not implied: this build plans one table, and the pair's task is
# expected to grow it (comma joins, join-order search, index-only scans, block indexes).
EXTENSION_SURFACE = (
    "single-table SELECT only",
    "no join planning, no index-only scan, no block-sparse index",
    "statistics come from the caller, not from a catalogue service",
)

