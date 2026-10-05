"""Logical rewriting and cost-based physical planning.

Pipeline: parse tree -> logical plan -> rewrites -> physical plan. The rewrites are deliberately few
and each one is observable in `explain`, because a rewrite nobody can see is a rewrite nobody can
debug:

  * predicate pushdown -- a filter is moved below the projection it feeds (never below an aggregate
    unless every referenced column is a grouping key); in a two-table plan every WHERE conjunct that
    references one input alone is pushed into that input's scan;
  * projection pruning -- a scan reads only the columns that survive to the top, per input;
  * constant folding -- comparisons between literals are decided at plan time;
  * limit pushdown -- a bounded sort keeps at most `limit` rows.

Physical selection is cost-based on the decisions that matter for this subset:

  * one table -- a filter that pins an indexed column to a literal becomes an `IndexScan`, otherwise
    the scan is full;
  * two tables -- an equality INNER JOIN is costed as hash-join building either side and, when the
    inner join key has an index, as index-nested-loop join; the cheapest physical plan wins, ties
    preferring hash-join and then the FROM-order build side.

Estimates use per-column distinct counts when the catalogue has them, and documented defaults when it
does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import PlanError, ValidationError
from .parser import (
    Aggregate,
    BoolOp,
    Column,
    Comparison,
    InList,
    IsNull,
    JoinClause,
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

# hash join: one build pass (hashing), one probe pass, plus one charge per emitted pair
HASH_BUILD_COST_PER_ROW = 0.5
HASH_PROBE_COST_PER_ROW = 0.3
JOIN_OUTPUT_COST_PER_ROW = 0.8


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
    left_table: str
    right_table: str
    left_key: str
    right_key: str


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
    qualified: bool = False


@dataclass(frozen=True, slots=True)
class PhysicalProject:
    input: object
    projections: tuple[Projection, ...]
    # True above a join: rows are keyed "table.column" and Column nodes carry their owning table.
    qualified: bool = False


@dataclass(frozen=True, slots=True)
class PhysicalAggregate:
    input: object
    group_by: tuple[str, ...]
    aggregates: tuple[Aggregate, ...]
    rows: int
    qualified: bool = False
    group_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhysicalSort:
    input: object
    keys: tuple[tuple[str, str | None, bool], ...]


@dataclass(frozen=True, slots=True)
class PhysicalLimit:
    input: object
    count: int


@dataclass(frozen=True, slots=True)
class HashJoin:
    """Equi-join. `build_side` names the hashed input; the other probes."""

    left: object
    right: object
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    build_side: str  # "left" | "right"
    rows: int


@dataclass(frozen=True, slots=True)
class IndexNestedLoopJoin:
    """Outer rows probe an index on the inner key. `outer_side` names the driving input.

    The inner side is never scanned: each outer row looks up `inner_index_column`, then the pushed
    inner predicates are applied to whatever the lookup returned. Cost scalars are stored explicitly
    so the cost model does not have to rediscover them.
    """

    outer: object
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    outer_side: str  # "left" | "right"
    inner_table: str
    inner_columns: tuple[str, ...]
    inner_index_column: str
    inner_predicates: tuple[object, ...]
    lookups: int
    matched_rows: float
    rows: int


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
            document["tables"] = [self.statement.table, *(join.table for join in self.statement.joins)]
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
        return {
            "operator": "join",
            "type": "inner",
            "left": describe_node(node.left),
            "right": describe_node(node.right),
            "leftTable": node.left_table,
            "rightTable": node.right_table,
            "leftKey": node.left_key,
            "rightKey": node.right_key,
            "joinOrder": [node.left_table, node.right_table],
        }
    if isinstance(node, FullScan):
        return {"operator": "full-scan", "table": node.table, "columns": list(node.columns), "rows": node.rows}
    if isinstance(node, IndexScan):
        return {"operator": "index-scan", "table": node.table, "column": node.column, "value": node.value, "columns": list(node.columns), "rows": node.rows}
    if isinstance(node, PhysicalFilter):
        return {"operator": "filter", "input": describe_node(node.input), "predicate": _render_predicate(node.predicate), "rows": node.rows}
    if isinstance(node, PhysicalProject):
        return {"operator": "project", "input": describe_node(node.input), "expressions": [_render_expression(item.expression) for item in node.projections]}
    if isinstance(node, PhysicalAggregate):
        group_by = list(node.group_labels or node.group_by)
        return {"operator": "aggregate", "input": describe_node(node.input), "groupBy": group_by, "functions": [item.function for item in node.aggregates], "rows": node.rows}
    if isinstance(node, PhysicalSort):
        return {"operator": "sort", "input": describe_node(node.input), "keys": [_sort_key_document(name, table, descending) for name, table, descending in node.keys]}
    if isinstance(node, PhysicalLimit):
        return {"operator": "limit", "input": describe_node(node.input), "count": node.count}
    if isinstance(node, HashJoin):
        return {
            "operator": "hash-join",
            "left": describe_node(node.left),
            "right": describe_node(node.right),
            "leftTable": node.left_table,
            "rightTable": node.right_table,
            "leftKey": node.left_key,
            "rightKey": node.right_key,
            "buildSide": node.build_side,
            "joinOrder": [node.left_table, node.right_table] if node.build_side == "left" else [node.right_table, node.left_table],
            "rows": node.rows,
            "cost": round(_estimate_cost(node), 3),
        }
    if isinstance(node, IndexNestedLoopJoin):
        outer_document = describe_node(node.outer)
        inner_document = {
            "operator": "index-lookup",
            "table": node.inner_table,
            "column": node.inner_index_column,
            "columns": list(node.inner_columns),
            "rows": node.rows,
        }
        return {
            "operator": "index-nested-loop-join",
            "left": outer_document if node.outer_side == "left" else inner_document,
            "right": inner_document if node.outer_side == "left" else outer_document,
            "outer": outer_document,
            "inner": inner_document,
            "leftTable": node.left_table,
            "rightTable": node.right_table,
            "leftKey": node.left_key,
            "rightKey": node.right_key,
            "outerSide": node.outer_side,
            "joinOrder": [node.left_table, node.right_table] if node.outer_side == "left" else [node.right_table, node.left_table],
            "rows": node.rows,
            "cost": round(_estimate_cost(node), 3),
        }
    raise ValidationError(f"cannot describe node: {type(node).__name__}")


def _sort_key_document(name: str, table: str | None, descending: bool) -> dict[str, object]:
    document = {"column": name, "descending": descending}
    if table is not None:
        document["table"] = table
    return document


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


def referenced_column_nodes(node: object) -> set[Column]:
    """Like `referenced_columns`, but qualified: every Column node an expression mentions."""
    if isinstance(node, Column):
        return {node}
    if isinstance(node, (Literal, Star)):
        return set()
    if isinstance(node, Aggregate):
        return referenced_column_nodes(node.argument)
    found: set[Column] = set()
    for attribute in ("left", "right", "operand"):
        child = getattr(node, attribute, None)
        if child is not None:
            found |= referenced_column_nodes(child)
    for attribute in ("operands", "values"):
        children = getattr(node, attribute, None) or ()
        for child in children:
            found |= referenced_column_nodes(child)
    return found


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
            notes.append(f"index scan on {table}.{column} (estimated {rows} of {info.rows} rows)")
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
        node = PhysicalSort(node, tuple(_sort_key(key) for key in statement.order_by))
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


# -- join planning -------------------------------------------------------------------------------
def _plan_join(statement: Select, catalog: Catalog) -> Plan:
    if len(statement.joins) > 1:
        raise PlanError("only one INNER JOIN is supported", joins=len(statement.joins))
    join = statement.joins[0]
    if join.table == statement.table:
        raise PlanError(f"self-join is not supported: {statement.table} appears on both sides")
    left_info = catalog.get(statement.table)
    right_info = catalog.get(join.table)
    notes: list[str] = []

    left_key, right_key = _check_join_condition(join, left_info, right_info)
    _check_join_columns(statement, left_info, right_info)

    conjuncts = _split_conjuncts(statement.where) if statement.where is not None else []
    left_predicates, right_predicates, residual = _classify_conjuncts(conjuncts, left_info, right_info)

    left_columns = _side_columns(statement, left_info, left_key, left_predicates, residual, notes)
    right_columns = _side_columns(statement, right_info, right_key, right_predicates, residual, notes)
    if left_predicates:
        notes.append(f"predicate pushdown: {len(left_predicates)} WHERE conjunct(s) moved into {left_info.name}")
    if right_predicates:
        notes.append(f"predicate pushdown: {len(right_predicates)} WHERE conjunct(s) moved into {right_info.name}")
    if residual:
        notes.append(f"filter: {len(residual)} conjunct(s) reference both inputs and stay above the join")

    left_predicate = _combine(left_predicates)
    right_predicate = _combine(right_predicates)
    # Side chains evaluate rows local to one table (plain column names), so pushed predicates are
    # localized; everything above the join runs on rows keyed "table.column" and is qualified.
    left_chain, left_rows = _build_side_chain(left_info, _localize(left_predicate, left_info), left_columns, notes)
    right_chain, right_rows = _build_side_chain(right_info, _localize(right_predicate, right_info), right_columns, notes)

    left_distinct = _filtered_distinct(left_info, left_key, left_rows, left_predicate)
    right_distinct = _filtered_distinct(right_info, right_key, right_rows, right_predicate)
    join_rows = max(0, int(left_rows * right_rows / max(left_distinct, right_distinct)))

    physical = _choose_join(
        left_chain,
        right_chain,
        left_info,
        right_info,
        left_key,
        right_key,
        left_rows,
        right_rows,
        left_distinct,
        right_distinct,
        join_rows,
        tuple(_localize(item, right_info) for item in right_predicates),
        tuple(_localize(item, left_info) for item in left_predicates),
        notes,
    )
    rows = join_rows

    residual_predicate = _qualify(_combine(residual), left_info, right_info)
    if residual_predicate is not None:
        merged = TableInfo(
            name=f"{left_info.name}+{right_info.name}",
            columns=tuple(dict.fromkeys(left_info.columns + right_info.columns)),
            rows=rows,
            distinct={**left_info.distinct, **right_info.distinct},
        )
        rows = max(1, int(rows * estimate_selectivity(residual_predicate, merged)))
        physical = PhysicalFilter(physical, residual_predicate, rows, qualified=True)

    projections = tuple(
        Projection(_qualify(item.expression, left_info, right_info), item.alias) for item in statement.projections
    )
    has_aggregate = any(isinstance(item.expression, Aggregate) for item in projections)
    if has_aggregate or statement.group_by:
        if not statement.group_by and not rows:
            rows = 1  # an aggregate over an empty join still emits one group
        aggregates = tuple(item.expression for item in projections if isinstance(item.expression, Aggregate))
        group_columns = [_qualify(column, left_info, right_info) for column in statement.group_by]
        estimated = 1 if not statement.group_by else max(1, int(rows * 0.5))
        physical = PhysicalAggregate(
            physical,
            tuple(_row_key(column) for column in group_columns),
            aggregates,
            estimated,
            qualified=True,
            group_labels=tuple(column.name for column in group_columns),
        )
        rows = estimated
        notes.append(f"aggregate: {len(statement.group_by)} grouping key(s), {len(aggregates)} aggregate(s)")

    physical = PhysicalProject(physical, projections, qualified=True)
    if statement.order_by:
        physical = PhysicalSort(physical, tuple(_sort_key(key) for key in statement.order_by))
        notes.append("sort: exhaustive sort of the projected rows")
    if statement.limit is not None:
        rows = min(rows, statement.limit)
        physical = PhysicalLimit(physical, statement.limit)
        notes.append(f"limit: at most {statement.limit} row(s) leave the plan")

    logical = _build_join_logical(statement, left_columns, right_columns, left_key, right_key)
    return Plan(
        statement=statement,
        logical=logical,
        physical=physical,
        estimated_rows=rows,
        estimated_cost=_estimate_cost(physical),
        notes=tuple(notes),
    )


def _sort_key(key: OrderKey) -> tuple[str, str | None, bool]:
    return key.column.name, key.column.table, key.descending


def _row_key(column: Column) -> str:
    return f"{column.table}.{column.name}" if column.table is not None else column.name


def _qualify(node: object, left_info: TableInfo, right_info: TableInfo) -> object:
    """Give every Column above a join its owning table (rows are keyed 'table.column' there)."""
    if isinstance(node, Column):
        owner = _resolve_side(node, left_info, right_info, "")
        return Column(node.name, owner)
    if isinstance(node, (Literal, Star)):
        return node
    if isinstance(node, Aggregate):
        return Aggregate(node.function, _qualify(node.argument, left_info, right_info), node.distinct)
    if isinstance(node, Comparison):
        return Comparison(_qualify(node.left, left_info, right_info), node.operator, _qualify(node.right, left_info, right_info))
    if isinstance(node, InList):
        return InList(
            _qualify(node.operand, left_info, right_info),
            tuple(_qualify(value, left_info, right_info) for value in node.values),
            node.negated,
        )
    if isinstance(node, IsNull):
        return IsNull(_qualify(node.operand, left_info, right_info), node.negated)
    if isinstance(node, Not):
        return Not(_qualify(node.operand, left_info, right_info))
    if isinstance(node, BoolOp):
        return BoolOp(node.operator, tuple(_qualify(operand, left_info, right_info) for operand in node.operands))
    return node


def _localize(node: object, info: TableInfo) -> object:
    """Drop table qualification from an expression evaluated against one table's local rows."""
    if node is None:
        return None
    if isinstance(node, Column):
        return Column(node.name)
    if isinstance(node, (Literal, Star)):
        return node
    if isinstance(node, Aggregate):
        return Aggregate(node.function, _localize(node.argument, info), node.distinct)
    if isinstance(node, Comparison):
        return Comparison(_localize(node.left, info), node.operator, _localize(node.right, info))
    if isinstance(node, InList):
        return InList(_localize(node.operand, info), tuple(_localize(value, info) for value in node.values), node.negated)
    if isinstance(node, IsNull):
        return IsNull(_localize(node.operand, info), node.negated)
    if isinstance(node, Not):
        return Not(_localize(node.operand, info))
    if isinstance(node, BoolOp):
        return BoolOp(node.operator, tuple(_localize(operand, info) for operand in node.operands))
    return node


def _check_join_condition(join: JoinClause, left_info: TableInfo, right_info: TableInfo) -> tuple[str, str]:
    on = join.on
    if not isinstance(on, Comparison) or on.operator != "=" or not isinstance(on.left, Column) or not isinstance(on.right, Column):
        raise PlanError("ON must be a single equality between one column from each input")
    left_side = _resolve_side(on.left, left_info, right_info, "ON")
    right_side = _resolve_side(on.right, left_info, right_info, "ON")
    if left_side == right_side:
        raise PlanError("join columns must come from different inputs, one from each table")
    return (on.left.name, on.right.name) if left_side == left_info.name else (on.right.name, on.left.name)


def _resolve_side(column: Column, left_info: TableInfo, right_info: TableInfo, where: str) -> str:
    """Which input a column belongs to; ambiguous/unknown/qualified-unknown are plan errors."""
    if column.table is not None:
        if column.table not in (left_info.name, right_info.name):
            raise PlanError(f"unknown table: {column.table}", in_clause=where, known=[left_info.name, right_info.name])
        info = left_info if column.table == left_info.name else right_info
        if column.name not in info.columns:
            raise PlanError(f"unknown column: {column.table}.{column.name}", in_clause=where, known=list(info.columns))
        return info.name
    owners = [info.name for info in (left_info, right_info) if column.name in info.columns]
    if not owners:
        raise PlanError(
            f"unknown column: {column.name}",
            in_clause=where,
            known=sorted(set(left_info.columns) | set(right_info.columns)),
        )
    if len(owners) == 2:
        raise PlanError(f"ambiguous column: {column.name} (present in both inputs)", in_clause=where)
    return owners[0]


def _check_join_columns(statement: Select, left_info: TableInfo, right_info: TableInfo) -> None:
    expressions: list[tuple[str, object]] = [
        (f"projection {index + 1}", item.expression) for index, item in enumerate(statement.projections)
    ]
    if statement.where is not None:
        expressions.append(("WHERE", statement.where))
    expressions.extend((f"GROUP BY {column.name}", column) for column in statement.group_by)
    expressions.extend((f"ORDER BY {key.column.name}", key.column) for key in statement.order_by)
    for where, expression in expressions:
        for column in referenced_column_nodes(expression):
            _resolve_side(column, left_info, right_info, where)


def _classify_conjuncts(
    conjuncts: list[object], left_info: TableInfo, right_info: TableInfo
) -> tuple[list[object], list[object], list[object]]:
    left: list[object] = []
    right: list[object] = []
    residual: list[object] = []
    for conjunct in conjuncts:
        sides = {
            _resolve_side(column, left_info, right_info, "WHERE")
            for column in referenced_column_nodes(conjunct)
        }
        if not sides:
            residual.append(conjunct)
        elif sides == {left_info.name}:
            left.append(conjunct)
        elif sides == {right_info.name}:
            right.append(conjunct)
        else:
            residual.append(conjunct)
    return left, right, residual


def _side_columns(
    statement: Select,
    info: TableInfo,
    key: str,
    predicates: list[object],
    residual: list[object],
    notes: list[str],
) -> tuple[str, ...]:
    """Columns one input must carry: its join key plus everything above the join that names it.

    Pushed conjuncts are evaluated on this side (so all their columns are read), and cross-side
    residual conjuncts still need this side's columns after the join.
    """
    expressions = [item.expression for item in statement.projections if not isinstance(item.expression, Star)]
    expressions.extend(predicates)
    expressions.extend(residual)
    expressions.extend(statement.group_by)
    expressions.extend(order_key.column for order_key in statement.order_by)

    owned: set[str] = {key}
    if any(isinstance(item.expression, Star) for item in statement.projections):
        owned |= set(info.columns)
    for expression in expressions:
        for column in referenced_column_nodes(expression):
            if column.table == info.name or (column.table is None and column.name in info.columns):
                owned.add(column.name)
    columns = tuple(column for column in info.columns if column in owned)
    if len(columns) != len(info.columns):
        notes.append(f"projection pruning: {info.name} scan reads {len(columns)} of {len(info.columns)} columns")
    return columns


def _combine(conjuncts: list[object]) -> object:
    if not conjuncts:
        return None
    if len(conjuncts) == 1:
        return conjuncts[0]
    return BoolOp("and", tuple(conjuncts))


def _filtered_distinct(info: TableInfo, key: str, filtered_rows: int, predicate: object) -> int:
    """Distinct count of the join key among the rows that survive the pushed predicates.

    Equality predicates pin the key to one value; otherwise the distinct count scales with the
    filtered fraction, never above the base count.
    """
    base = max(1, info.distinct.get(key, max(1, info.rows)))
    fraction = 1.0 if info.rows == 0 else filtered_rows / info.rows
    distinct = max(1, int(base * fraction))
    for conjunct in _split_conjuncts(predicate):
        pinned = _pinned_value(conjunct)
        if pinned is not None and pinned[0] == key:
            return 1
    return min(max(1, filtered_rows), distinct)


def _build_side_chain(
    info: TableInfo, predicate: object, columns: tuple[str, ...], notes: list[str]
) -> tuple[object, int]:
    """Full/index scan plus the pushed filters for one input; returns (chain, filtered rows)."""
    node = _choose_scan(info.name, info, predicate, columns, notes)
    rows = int(getattr(node, "rows"))
    consumed: tuple[str, object] | None = None
    if isinstance(node, IndexScan):
        consumed = (node.column, node.value)
    pending = [
        conjunct
        for conjunct in (_split_conjuncts(predicate) if predicate is not None else [])
        if not (consumed is not None and _pinned_value(conjunct) == consumed)
    ]
    for conjunct in pending:
        rows = max(1, int(rows * estimate_selectivity(conjunct, info)))
        node = PhysicalFilter(node, conjunct, rows)
    return node, rows


def _choose_join(
    left_chain: object,
    right_chain: object,
    left_info: TableInfo,
    right_info: TableInfo,
    left_key: str,
    right_key: str,
    left_rows: int,
    right_rows: int,
    left_distinct: int,
    right_distinct: int,
    join_rows: int,
    right_inner_predicates: tuple[object, ...],
    left_inner_predicates: tuple[object, ...],
    notes: list[str],
) -> object:
    left_cost = _estimate_cost(left_chain)
    right_cost = _estimate_cost(right_chain)

    candidates: list[tuple[str, object, float]] = []
    # Candidate order is the documented tie-break: hash-join first, and within hash the FROM order.
    hash_build_left_cost = (
        left_cost
        + right_cost
        + HASH_BUILD_COST_PER_ROW * left_rows
        + HASH_PROBE_COST_PER_ROW * right_rows
        + JOIN_OUTPUT_COST_PER_ROW * join_rows
    )
    candidates.append((
        f"hash-join building {left_info.name}",
        HashJoin(left_chain, right_chain, left_info.name, right_info.name, left_key, right_key, "left", join_rows),
        hash_build_left_cost,
    ))
    hash_build_right_cost = (
        left_cost
        + right_cost
        + HASH_BUILD_COST_PER_ROW * right_rows
        + HASH_PROBE_COST_PER_ROW * left_rows
        + JOIN_OUTPUT_COST_PER_ROW * join_rows
    )
    candidates.append((
        f"hash-join building {right_info.name}",
        HashJoin(left_chain, right_chain, left_info.name, right_info.name, left_key, right_key, "right", join_rows),
        hash_build_right_cost,
    ))

    if right_key in right_info.indexes:
        matched = left_rows * right_rows / max(1, right_distinct)
        cost = (
            left_cost
            + left_rows * INDEX_COST
            + matched * SCAN_COST_PER_ROW
            + JOIN_OUTPUT_COST_PER_ROW * join_rows
        )
        candidates.append((
            f"index-nested-loop join, {left_info.name} outer / {right_info.name} indexed",
            IndexNestedLoopJoin(
                outer=left_chain,
                left_table=left_info.name,
                right_table=right_info.name,
                left_key=left_key,
                right_key=right_key,
                outer_side="left",
                inner_table=right_info.name,
                inner_columns=_scan_columns(right_chain),
                inner_index_column=right_key,
                inner_predicates=right_inner_predicates,
                lookups=left_rows,
                matched_rows=matched,
                rows=join_rows,
            ),
            cost,
        ))
    if left_key in left_info.indexes:
        matched = right_rows * left_rows / max(1, left_distinct)
        cost = (
            right_cost
            + right_rows * INDEX_COST
            + matched * SCAN_COST_PER_ROW
            + JOIN_OUTPUT_COST_PER_ROW * join_rows
        )
        candidates.append((
            f"index-nested-loop join, {right_info.name} outer / {left_info.name} indexed",
            IndexNestedLoopJoin(
                outer=right_chain,
                left_table=left_info.name,
                right_table=right_info.name,
                left_key=left_key,
                right_key=right_key,
                outer_side="right",
                inner_table=left_info.name,
                inner_columns=_scan_columns(left_chain),
                inner_index_column=left_key,
                inner_predicates=left_inner_predicates,
                lookups=right_rows,
                matched_rows=matched,
                rows=join_rows,
            ),
            cost,
        ))

    label, node, cost = candidates[0]
    for candidate_label, candidate_node, candidate_cost in candidates[1:]:
        if candidate_cost < cost:
            label, node, cost = candidate_label, candidate_node, candidate_cost
    notes.append("join candidates: " + "; ".join(f"{name} ~ {value:.3f}" for name, _, value in candidates))
    notes.append(f"join choice: {label} (estimated {join_rows} output rows)")
    return node


def _scan_columns(chain: object) -> tuple[str, ...]:
    while isinstance(chain, PhysicalFilter):
        chain = chain.input
    return tuple(getattr(chain, "columns", ()))


def _build_join_logical(
    statement: Select,
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
    left_key: str,
    right_key: str,
) -> object:
    left: object = LogicalScan(statement.table, left_columns)
    right: object = LogicalScan(statement.joins[0].table, right_columns)
    node: object = LogicalJoin(left, right, statement.table, statement.joins[0].table, left_key, right_key)
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
    if isinstance(node, (HashJoin, IndexNestedLoopJoin)):
        return int(node.rows)
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
        left_cost, right_cost = _estimate_cost(node.left), _estimate_cost(node.right)
        build = rows_produced(node.left) if node.build_side == "left" else rows_produced(node.right)
        probe = rows_produced(node.right) if node.build_side == "left" else rows_produced(node.left)
        return (
            left_cost
            + right_cost
            + HASH_BUILD_COST_PER_ROW * build
            + HASH_PROBE_COST_PER_ROW * probe
            + JOIN_OUTPUT_COST_PER_ROW * node.rows
        )
    if isinstance(node, IndexNestedLoopJoin):
        inner_filter_cost = node.matched_rows * FILTER_COST_PER_ROW * len(node.inner_predicates)
        return (
            _estimate_cost(node.outer)
            + node.lookups * INDEX_COST
            + node.matched_rows * SCAN_COST_PER_ROW
            + inner_filter_cost
            + JOIN_OUTPUT_COST_PER_ROW * node.rows
        )
    raise ValidationError(f"cannot cost node: {type(node).__name__}")


def explain(plan_result: Plan) -> dict[str, object]:
    return plan_result.to_document()


# The extension surface is stated, not implied: this build plans one table or an equality INNER JOIN
# of exactly two, and later tasks grow it (three-plus tables and join reordering, outer joins,
# non-equi joins, index-only scans, block indexes).
EXTENSION_SURFACE = (
    "single-table SELECT, or exactly two tables joined by one equality INNER JOIN",
    "no outer joins, no non-equi joins, no self joins, no three-or-more-table plans",
    "no join ordering beyond the hash build side and one index-nested-loop alternative",
    "no index-only scan, no block-sparse index",
    "statistics come from the caller, not from a catalogue service",
)
