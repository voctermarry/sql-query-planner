"""Logical rewriting and cost-based physical planning.

Pipeline: parse tree -> logical plan -> rewrites -> physical plan. The rewrites are deliberately few
and each one is observable in `explain`, because a rewrite nobody can see is a rewrite nobody can
debug:

  * predicate pushdown -- a filter is moved below the projection it feeds (never below an aggregate
    unless every referenced column is a grouping key);
  * projection pruning -- a scan reads only the columns that survive to the top;
  * constant folding -- comparisons between literals are decided at plan time;
  * limit pushdown -- a bounded sort keeps at most `limit` rows.

Physical selection is cost-based on the decisions that matter for this subset: a filter that pins an
indexed column to a literal becomes an `IndexScan`, otherwise the scan is full; a two-table equi-join
becomes a hash join (build side chosen by cost) or, when a join key is indexed, an index nested loop.
Estimates use per-column distinct counts when the catalogue has them, and documented defaults when it
does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from .errors import PlanError, ValidationError
from .parser import (
    Aggregate,
    BoolOp,
    Column,
    Comparison,
    InList,
    IsNull,
    Join,
    Literal,
    Not,
    OrderKey,
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
# join costs: building a hash table is more expensive per row than probing it, so the smaller
# (filtered) side wins the build; an index nested loop pays INDEX_COST per outer row instead.
HASH_BUILD_COST_PER_ROW = 1.0
HASH_PROBE_COST_PER_ROW = 0.5


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


@dataclass(frozen=True, slots=True)
class LogicalJoin:
    left: object
    right: object
    left_key: str
    right_key: str


# -- physical nodes ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FullScan:
    table: str
    columns: tuple[str, ...]
    rows: int
    # qualify: emit row keys as "table.column" (join plans need both inputs in one row)
    qualify: bool = False


@dataclass(frozen=True, slots=True)
class IndexScan:
    table: str
    column: str
    value: object
    columns: tuple[str, ...]
    rows: int
    qualify: bool = False


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
class HashJoin:
    """Equi-join: hash the build side's key, probe with the other side.

    `left_key`/`right_key` are qualified ("table.column") because both inputs share one row
    downstream. `columns` is the qualified output layout, left input first, so every physical
    join algorithm hands the same shape upward.
    """

    left: object
    right: object
    left_key: str
    right_key: str
    build: str  # "left" | "right"
    columns: tuple[str, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class IndexNestedLoop:
    """Equi-join: scan the outer side, look each key up in the inner table's index."""

    outer: object
    outer_is_left: bool
    inner_table: str
    inner_key: str  # unqualified: the indexed column on the inner table
    outer_key: str  # qualified key in the outer rows
    inner_columns: tuple[str, ...]  # unqualified columns read from the inner table
    inner_predicate: object | None  # pushed-down conjuncts applied to each fetched inner row
    columns: tuple[str, ...]
    rows: int
    matches: int  # estimated inner rows per lookup, for costing


@dataclass(frozen=True, slots=True)
class Plan:
    statement: Select
    logical: object
    physical: object
    estimated_rows: int
    estimated_cost: float
    notes: tuple[str, ...] = ()

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "table": self.statement.table,
            "estimatedRows": self.estimated_rows,
            "estimatedCost": round(self.estimated_cost, 3),
            "notes": list(self.notes),
            "logical": describe_node(self.logical),
            "physical": describe_node(self.physical),
        }
        if self.statement.joins:
            summary = join_summary(self.physical)
            if summary is not None:
                document["join"] = summary
        return document


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
    if isinstance(node, LogicalJoin):
        return {"operator": "join", "leftKey": node.left_key, "rightKey": node.right_key, "left": describe_node(node.left), "right": describe_node(node.right)}
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
    if isinstance(node, HashJoin):
        return {
            "operator": "hash-join",
            "build": node.build,
            "leftKey": node.left_key,
            "rightKey": node.right_key,
            "columns": list(node.columns),
            "rows": node.rows,
            "left": describe_node(node.left),
            "right": describe_node(node.right),
        }
    if isinstance(node, IndexNestedLoop):
        return {
            "operator": "index-nested-loop",
            "innerTable": node.inner_table,
            "innerKey": node.inner_key,
            "outerKey": node.outer_key,
            "rows": node.rows,
            "outer": describe_node(node.outer),
        }
    raise ValidationError(f"cannot describe node: {type(node).__name__}")


def _render_expression(node: object) -> dict[str, object]:
    if isinstance(node, Column):
        return {"column": node.name} | ({"table": node.table} if node.table else {})
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
    if statement.joins:
        return _plan_join(statement, catalog)
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
    if isinstance(node, (FullScan, IndexScan, PhysicalFilter, PhysicalAggregate, HashJoin, IndexNestedLoop)):
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
    if isinstance(node, HashJoin):
        build = node.left if node.build == "left" else node.right
        probe = node.right if node.build == "left" else node.left
        return (
            _estimate_cost(node.left)
            + _estimate_cost(node.right)
            + HASH_BUILD_COST_PER_ROW * rows_produced(build)
            + HASH_PROBE_COST_PER_ROW * rows_produced(probe)
        )
    if isinstance(node, IndexNestedLoop):
        return _estimate_cost(node.outer) + rows_produced(node.outer) * (INDEX_COST + SCAN_COST_PER_ROW * node.matches)
    raise ValidationError(f"cannot cost node: {type(node).__name__}")


# -- join planning -------------------------------------------------------------------------------
def _column_nodes(node: object) -> set[Column]:
    """Every Column an expression mentions, qualifiers kept (unlike referenced_columns)."""
    if isinstance(node, Column):
        return {node}
    if node is None or isinstance(node, (Literal, Star)):
        return set()
    if isinstance(node, Aggregate):
        return _column_nodes(node.argument)
    found: set[Column] = set()
    for attribute in ("left", "right", "operand"):
        child = getattr(node, attribute, None)
        if child is not None:
            found |= _column_nodes(child)
    for attribute in ("operands", "values"):
        for child in getattr(node, attribute, None) or ():
            found |= _column_nodes(child)
    return found


def _and_of(conjuncts: list[object]) -> object | None:
    if not conjuncts:
        return None
    if len(conjuncts) == 1:
        return conjuncts[0]
    return BoolOp("and", tuple(conjuncts))


def _qualified_name(column: Column) -> str:
    return f"{column.table}.{column.name}" if column.table else column.name


def _qualify_expression(node: object, resolve: object, clause: str) -> object:
    """Rewrite every Column in an expression through `resolve` (which attaches its table)."""
    if node is None or isinstance(node, (Literal, Star)):
        return node
    if isinstance(node, Column):
        return resolve(node, clause)
    if isinstance(node, Aggregate):
        return replace(node, argument=_qualify_expression(node.argument, resolve, clause))
    if isinstance(node, Comparison):
        return Comparison(_qualify_expression(node.left, resolve, clause), node.operator, _qualify_expression(node.right, resolve, clause))
    if isinstance(node, InList):
        return InList(_qualify_expression(node.operand, resolve, clause), tuple(_qualify_expression(value, resolve, clause) for value in node.values), node.negated)
    if isinstance(node, IsNull):
        return IsNull(_qualify_expression(node.operand, resolve, clause), node.negated)
    if isinstance(node, Not):
        return Not(_qualify_expression(node.operand, resolve, clause))
    if isinstance(node, BoolOp):
        return BoolOp(node.operator, tuple(_qualify_expression(operand, resolve, clause) for operand in node.operands))
    raise ValidationError(f"cannot qualify expression: {type(node).__name__}")


def _join_side(table: str, info: TableInfo, predicate: object, columns: tuple[str, ...], notes: list[str]) -> tuple[object, int]:
    """One join input: cost-based scan plus its pushed-down conjuncts, rows qualified."""
    scan = replace(_choose_scan(table, info, predicate, columns, notes), qualify=True)
    rows = int(scan.rows)
    consumed: tuple[str, object] | None = None
    if isinstance(scan, IndexScan):
        consumed = (scan.column, scan.value)
        notes.append(f"predicate pushdown: '{scan.column} = {scan.value!r}' is answered by the index on {table}")
    node: object = scan
    pending: list[object] = []
    for conjunct in _split_conjuncts(predicate) if predicate is not None else []:
        if consumed is not None and _pinned_value(conjunct) == consumed:
            continue
        pending.append(conjunct)
    for conjunct in pending:
        rows = max(1, int(rows * estimate_selectivity(conjunct, info)))
        node = PhysicalFilter(node, conjunct, rows)
    if pending:
        notes.append(f"filter: {len(pending)} residual predicate(s) on {table} evaluated row by row")
    return node, rows


def _join_sort_keys(statement: Select) -> tuple[tuple[str, bool], ...]:
    """ORDER BY runs above the projection, so keys must name projection output labels."""
    labels: dict[str, str] = {}
    for projection in statement.projections:
        if isinstance(projection.expression, Column):
            labels[_qualified_name(projection.expression)] = projection.alias or projection.expression.name
    keys: list[tuple[str, bool]] = []
    for key in statement.order_by:
        qualified = _qualified_name(key.column)
        keys.append((labels.get(qualified, qualified), key.descending))
    return tuple(keys)


def _plan_join(statement: Select, catalog: Catalog) -> Plan:
    if len(statement.joins) > 1:
        raise PlanError("only one INNER JOIN per query is supported", joins=len(statement.joins))
    join = statement.joins[0]
    left_name, right_name = statement.table, join.table
    if left_name == right_name:
        raise PlanError("self joins are not supported", table=left_name)
    left_info = catalog.get(left_name)
    right_info = catalog.get(right_name)
    notes: list[str] = []

    def resolve(column: Column, clause: str) -> Column:
        if column.table is not None:
            if column.table not in (left_name, right_name):
                raise PlanError(f"unknown table: {column.table}", in_clause=clause, known=[left_name, right_name])
            info = left_info if column.table == left_name else right_info
            if column.name not in info.columns:
                raise PlanError(f"unknown column: {column.name}", in_clause=clause, table=column.table, known=list(info.columns))
            return column
        in_left = column.name in left_info.columns
        in_right = column.name in right_info.columns
        if in_left and in_right:
            raise PlanError(f"ambiguous column: {column.name}", in_clause=clause, tables=[left_name, right_name])
        if not in_left and not in_right:
            raise PlanError(f"unknown column: {column.name}", in_clause=clause, known=sorted(set(left_info.columns) | set(right_info.columns)))
        return Column(column.name, left_name if in_left else right_name)

    condition = join.condition
    if not (
        isinstance(condition, Comparison)
        and condition.operator == "="
        and isinstance(condition.left, Column)
        and isinstance(condition.right, Column)
    ):
        raise PlanError("join condition must be one equality between two columns", clause="ON")
    on_left = resolve(condition.left, "ON")
    on_right = resolve(condition.right, "ON")
    if on_left.table == on_right.table:
        raise PlanError("join keys must come from different inputs", clause="ON")
    left_key, right_key = (on_left, on_right) if on_left.table == left_name else (on_right, on_left)

    resolved = _resolve_join_statement(statement, resolve, left_info, right_info, left_key, right_key)

    # predicate pushdown: single-side WHERE conjuncts move to that side's scan, the rest
    # (anything mentioning both inputs) stays as a filter above the join.
    conjuncts = _split_conjuncts(resolved.where) if resolved.where is not None else []
    left_conjuncts: list[object] = []
    right_conjuncts: list[object] = []
    residual: list[object] = []
    for conjunct in conjuncts:
        tables = {column.table for column in _column_nodes(conjunct)}
        if tables == {left_name}:
            left_conjuncts.append(conjunct)
        elif tables == {right_name}:
            right_conjuncts.append(conjunct)
        else:
            residual.append(conjunct)
    left_predicate = _and_of(left_conjuncts)
    right_predicate = _and_of(right_conjuncts)
    if left_conjuncts:
        notes.append(f"predicate pushdown: {len(left_conjuncts)} conjunct(s) applied to {left_name}")
    if right_conjuncts:
        notes.append(f"predicate pushdown: {len(right_conjuncts)} conjunct(s) applied to {right_name}")

    # projection pruning, per side: each scan reads only what the query needs from that table.
    used: set[Column] = set()
    for projection in resolved.projections:
        used |= _column_nodes(projection.expression)
    for conjunct in conjuncts:
        used |= _column_nodes(conjunct)
    used |= set(resolved.group_by)
    used |= {key.column for key in resolved.order_by}
    needed_left = {column.name for column in used if column.table == left_name} | {left_key.name}
    needed_right = {column.name for column in used if column.table == right_name} | {right_key.name}
    scan_left = tuple(column for column in left_info.columns if column in needed_left)
    scan_right = tuple(column for column in right_info.columns if column in needed_right)
    if len(scan_left) != len(left_info.columns):
        notes.append(f"projection pruning: {left_name} scan reads {len(scan_left)} of {len(left_info.columns)} columns")
    if len(scan_right) != len(right_info.columns):
        notes.append(f"projection pruning: {right_name} scan reads {len(scan_right)} of {len(right_info.columns)} columns")

    left_node, left_rows = _join_side(left_name, left_info, left_predicate, scan_left, notes)
    right_node, right_rows = _join_side(right_name, right_info, right_predicate, scan_right, notes)

    qualified_left_key = _qualified_name(left_key)
    qualified_right_key = _qualified_name(right_key)
    output_columns = tuple(f"{left_name}.{column}" for column in scan_left) + tuple(f"{right_name}.{column}" for column in scan_right)

    # join cardinality from the filtered estimates and the catalogue's distinct counts;
    # a key without statistics is treated as unique (distinct = table rows).
    distinct_left = left_info.distinct.get(left_key.name) or max(left_info.rows, 1)
    distinct_right = right_info.distinct.get(right_key.name) or max(right_info.rows, 1)
    if left_rows == 0 or right_rows == 0:
        join_rows = 0
    else:
        join_rows = max(1, int(left_rows * right_rows / max(distinct_left, distinct_right, 1)))

    # candidates in preference order: hash join before index nested loop, and inside hash the
    # FROM-order build side first -- iteration keeps the first candidate on an exact cost tie.
    candidates: list[tuple[str, object]] = [
        (
            "hash-join build=left",
            HashJoin(left_node, right_node, qualified_left_key, qualified_right_key, "left", output_columns, join_rows),
        ),
        (
            "hash-join build=right",
            HashJoin(left_node, right_node, qualified_left_key, qualified_right_key, "right", output_columns, join_rows),
        ),
    ]
    if right_key.name in right_info.indexes:
        matches = max(1, int(right_rows / max(distinct_right, 1)))
        candidates.append(
            (
                "index-nested-loop outer=left",
                IndexNestedLoop(left_node, True, right_name, right_key.name, qualified_left_key, scan_right, right_predicate, output_columns, join_rows, matches),
            )
        )
    if left_key.name in left_info.indexes:
        matches = max(1, int(left_rows / max(distinct_left, 1)))
        candidates.append(
            (
                "index-nested-loop outer=right",
                IndexNestedLoop(right_node, False, left_name, left_key.name, qualified_right_key, scan_left, left_predicate, output_columns, join_rows, matches),
            )
        )

    best_label = ""
    best: object | None = None
    best_cost = 0.0
    for label, candidate in candidates:
        cost = _estimate_cost(candidate)
        notes.append(f"join candidate {label}: estimated cost {round(cost, 3)}")
        if best is None or cost < best_cost:
            best_label, best, best_cost = label, candidate, cost
    node = best
    rows = join_rows
    notes.append(f"join: {best_label} selected (estimated {join_rows} rows, cost {round(best_cost, 3)})")

    if residual:
        # cross-table conjuncts have no single catalogue; documented defaults carry the estimate
        merged = TableInfo(f"{left_name}+{right_name}", (), rows)
        for conjunct in residual:
            rows = max(1, int(rows * estimate_selectivity(conjunct, merged)))
            node = PhysicalFilter(node, conjunct, rows)
        notes.append(f"filter: {len(residual)} cross-table predicate(s) evaluated after the join")

    has_aggregate = any(isinstance(item.expression, Aggregate) for item in resolved.projections)
    if has_aggregate and not resolved.group_by:
        notes.append("global aggregate: no GROUP BY, so every row folds into one group")
    if has_aggregate or resolved.group_by:
        aggregates = tuple(item.expression for item in resolved.projections if isinstance(item.expression, Aggregate))
        estimated = 1 if not resolved.group_by else max(1, int(rows * 0.5))
        node = PhysicalAggregate(node, tuple(_qualified_name(column) for column in resolved.group_by), aggregates, estimated)
        rows = estimated
        notes.append(f"aggregate: {len(resolved.group_by)} grouping key(s), {len(aggregates)} aggregate(s)")

    node = PhysicalProject(node, resolved.projections)
    if resolved.order_by:
        node = PhysicalSort(node, _join_sort_keys(resolved))
        notes.append("sort: exhaustive sort of the projected rows")
    if resolved.limit is not None:
        rows = min(rows, resolved.limit)
        node = PhysicalLimit(node, resolved.limit)
        notes.append(f"limit: at most {resolved.limit} row(s) leave the plan")

    return Plan(
        statement=resolved,
        logical=_build_join_logical(resolved, left_info, right_info, scan_left, scan_right, left_predicate, right_predicate, residual, qualified_left_key, qualified_right_key),
        physical=node,
        estimated_rows=rows,
        estimated_cost=_estimate_cost(node),
        notes=tuple(notes),
    )


def _resolve_join_statement(
    statement: Select,
    resolve: object,
    left_info: TableInfo,
    right_info: TableInfo,
    left_key: Column,
    right_key: Column,
) -> Select:
    """One pass that qualifies every column and expands '*' against both inputs.

    A star over a join emits the left table's columns then the right's; a name both inputs
    share is aliased to its qualified form so the output record cannot collide.
    """
    projections: list[Projection] = []
    for index, item in enumerate(statement.projections):
        if isinstance(item.expression, Star):
            for info, other in ((left_info, right_info), (right_info, left_info)):
                for column in info.columns:
                    alias = f"{info.name}.{column}" if column in other.columns else None
                    projections.append(Projection(Column(column, info.name), alias=alias))
        else:
            projections.append(Projection(_qualify_expression(item.expression, resolve, f"projection {index + 1}"), item.alias))
    where = _qualify_expression(statement.where, resolve, "WHERE") if statement.where is not None else None
    group_by = tuple(resolve(column, "GROUP BY") for column in statement.group_by)
    order_by = tuple(OrderKey(resolve(key.column, "ORDER BY"), key.descending) for key in statement.order_by)
    normalized = Join(table=right_info.name, condition=Comparison(left_key, "=", right_key))
    return replace(statement, projections=tuple(projections), where=where, group_by=group_by, order_by=order_by, joins=(normalized,))


def _build_join_logical(
    statement: Select,
    left_info: TableInfo,
    right_info: TableInfo,
    scan_left: tuple[str, ...],
    scan_right: tuple[str, ...],
    left_predicate: object,
    right_predicate: object,
    residual: list[object],
    left_key: str,
    right_key: str,
) -> object:
    left: object = LogicalScan(left_info.name, scan_left)
    if left_predicate is not None:
        left = LogicalFilter(left, left_predicate)
    right: object = LogicalScan(right_info.name, scan_right)
    if right_predicate is not None:
        right = LogicalFilter(right, right_predicate)
    node: object = LogicalJoin(left, right, left_key, right_key)
    if residual:
        node = LogicalFilter(node, _and_of(residual))
    has_aggregate = any(isinstance(item.expression, Aggregate) for item in statement.projections)
    if has_aggregate or statement.group_by:
        aggregates = tuple(item.expression for item in statement.projections if isinstance(item.expression, Aggregate))
        node = LogicalAggregate(node, tuple(_qualified_name(column) for column in statement.group_by), aggregates)
    node = LogicalProject(node, statement.projections)
    if statement.order_by:
        node = LogicalSort(node, _join_sort_keys(statement))
    if statement.limit is not None:
        node = LogicalLimit(node, statement.limit)
    return node


def _bottom_table(node: object) -> str | None:
    """The table at the bottom of a scan/filter chain (the side a join candidate reads)."""
    while hasattr(node, "input"):
        node = node.input
    return getattr(node, "table", None)


def join_summary(physical: object) -> dict[str, object] | None:
    """The join operator inside a physical plan, flattened for plan JSON and reconcile."""
    node = physical
    while node is not None:
        if isinstance(node, HashJoin):
            build_side = node.left if node.build == "left" else node.right
            probe_side = node.right if node.build == "left" else node.left
            return {
                "operator": "hash-join",
                "leftKey": node.left_key,
                "rightKey": node.right_key,
                "estimatedRows": node.rows,
                "build": node.build,
                "order": [_bottom_table(build_side), _bottom_table(probe_side)],
            }
        if isinstance(node, IndexNestedLoop):
            return {
                "operator": "index-nested-loop",
                "leftKey": f"{node.inner_table}.{node.inner_key}" if not node.outer_is_left else node.outer_key,
                "rightKey": node.outer_key if not node.outer_is_left else f"{node.inner_table}.{node.inner_key}",
                "estimatedRows": node.rows,
                "order": [_bottom_table(node.outer), node.inner_table],
            }
        node = getattr(node, "input", None)
    return None


def explain(plan_result: Plan) -> dict[str, object]:
    return plan_result.to_document()


# The extension surface is stated, not implied: this build plans one equi-join between two
# tables, and the pair's task is expected to grow it (multi-way joins, outer joins, index-only
# scans, block indexes).
EXTENSION_SURFACE = (
    "single-table SELECT and two-table equi INNER JOIN (hash join, or index nested loop when an index exists)",
    "no multi-way joins, no self joins, no outer joins, no index-only scan, no block-sparse index",
    "statistics come from the caller, not from a catalogue service",
)

