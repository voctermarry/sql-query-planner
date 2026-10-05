"""Deterministic physical-plan equivalence tests.

For one fixed SQL statement and one fixed dataset, the planner may legitimately choose different
physical plans: a full scan or an index scan, a hash join building either side, or an
index-nested-loop join. Whatever it chooses, the executed columns and rows -- including their
order and their duplicates -- must be identical. This suite proves that for three levers, each
driving one documented plan dimension:

  * index declarations  -> full-scan vs index-scan, and hash-join vs index-nested-loop-join;
  * distinct statistics -> hash-join build side (stale statistics must never change results);
  * data scale          -> the datasets are sized so each lever actually flips the choice.

Every table is generated from a fixed seed by `random.Random`, so a case reproduces byte-for-byte
from its seed alone; a guard test proves the generators are deterministic and that the edge
shapes the cases rely on (duplicate join keys, NULL keys, missing fields, empty tables,
filtered-empty sets) are really present. Only documented JSON scalar values (integers, decimals,
strings, null) appear in the data, and every statement stays inside the documented SQL subset.

A case is invalid -- and fails loudly -- when its two catalogues produce the *same* physical
plan: comparing a plan to itself would prove nothing. When results diverge, the failure message
dumps the SQL, the seeds, the table data, the index/statistics configuration, both plan
summaries and both results, so the offending operator can be located without rerunning anything.

Results are compared as ordered lists of rows, never as sets: without ORDER BY the baseline
promises logical-left original order then right original order for every physical algorithm, and
only a row-by-row comparison can catch a reordered or duplicated row.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from dataclasses import dataclass, field

from sqlplan.cli import EXIT_OK, main
from sqlplan.executor import Table, execute
from sqlplan.parser import parse
from sqlplan.planner import Catalog, TableInfo, describe_node, plan

# -- deterministic datasets -----------------------------------------------------------------------
# One seed per dataset; regenerating from these seeds must reproduce the tables exactly (a test
# below asserts that). Sizes are chosen so the cost model flips each documented choice: 60 rows
# with 3 regions makes an index scan strictly cheaper than a full scan; 80 vs 20 rows lets a
# stale distinct statistic flip the hash build side; 4 vs 200 rows lets an index on the join key
# make index-nested-loop cheaper than hashing.
ORDERS_SEED = 101
BUILD_LEFT_SEED = 202
BUILD_RIGHT_SEED = 203
JOIN_ORDERS_SEED = 303

SEEDS = {
    "orders": ORDERS_SEED,
    "build_left": BUILD_LEFT_SEED,
    "build_right": BUILD_RIGHT_SEED,
    "join_orders": JOIN_ORDERS_SEED,
}


def generate_orders(seed: int, count: int) -> list[dict[str, object]]:
    """Orders with duplicate and NULL join keys, and a `note` that is sometimes NULL, sometimes absent."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for identifier in range(1, count + 1):
        row: dict[str, object] = {
            "id": identifier,
            "cid": rng.choice([1, 2, 3, 4, 5, None]),
            "region": rng.choice(["eu", "us", "ap"]),
            "amount": round(rng.uniform(1.0, 500.0), 2),
        }
        if rng.random() < 0.8:
            row["note"] = rng.choice(["x", "y", "z", None])
        rows.append(row)
    return rows


def generate_customers(seed: int, count: int) -> list[dict[str, object]]:
    """Customers with duplicate and NULL join keys, and a `name` that is sometimes absent."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _ in range(count):
        row: dict[str, object] = {"cid": rng.choice([1, 2, 3, 4, 5, 6, None])}
        if rng.random() < 0.85:
            row["name"] = rng.choice(["a", "b", "c", "d"])
        rows.append(row)
    return rows


ORDERS = generate_orders(ORDERS_SEED, 60)
BUILD_ORDERS = generate_orders(BUILD_LEFT_SEED, 80)
BUILD_CUSTOMERS = generate_customers(BUILD_RIGHT_SEED, 20)
JOIN_ORDERS = generate_orders(JOIN_ORDERS_SEED, 200)
# Hand-written on purpose: one row missing `name`, one NULL join key, duplicate keys across the join.
JOIN_CUSTOMERS = [{"cid": 1, "name": "a"}, {"cid": 2, "name": "b"}, {"cid": 3}, {"cid": None, "name": "d"}]

CUSTOMER_SCHEMA = ("cid", "name")


# -- case definitions -------------------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogVariant:
    """One catalogue view over the same rows: index declarations and (stale) distinct overrides."""

    indexes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    distinct: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_document(self) -> dict[str, object]:
        return {"indexes": {k: list(v) for k, v in self.indexes.items()}, "distinct": self.distinct}


@dataclass(frozen=True)
class EquivalenceCase:
    """One SQL statement planned through two catalogues; the physical plans must differ as
    `expect` describes and the executed results must be byte-identical.

    `expect` entries: "accessPath:<table>" (the table's leaf scan operator differs),
    "buildSide" (both plans are hash joins building opposite sides), "joinOperator" (the join
    operator itself differs). `schemas` declares columns for tables too empty to derive them from.
    """

    name: str
    sql: str
    tables: dict[str, list[dict[str, object]]]
    variant_a: CatalogVariant
    variant_b: CatalogVariant
    expect: tuple[str, ...]
    schemas: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _derive_schema(rows: list[dict[str, object]]) -> tuple[list[str], dict[str, int]]:
    """Columns and per-column distinct counts, derived from the rows exactly as the CLI does."""
    columns = sorted({key for row in rows for key in row})
    return columns, {column: len({row.get(column) for row in rows}) for column in columns}


def _build_catalog(case: EquivalenceCase, variant: CatalogVariant) -> Catalog:
    catalog = Catalog()
    for name, rows in case.tables.items():
        columns, distinct = _derive_schema(rows)
        if name in case.schemas:  # an empty table has no rows to derive a schema from
            columns = list(case.schemas[name])
            distinct = {column: distinct.get(column, 0) for column in columns}
        distinct.update(variant.distinct.get(name, {}))
        catalog.add(TableInfo(name, tuple(columns), len(rows), distinct, tuple(variant.indexes.get(name, ()))))
    return catalog


def _build_tables(case: EquivalenceCase) -> dict[str, Table]:
    tables: dict[str, Table] = {}
    for name, rows in case.tables.items():
        columns, _ = _derive_schema(rows)
        if name in case.schemas:
            columns = list(case.schemas[name])
        tables[name] = Table(name, tuple(columns), [dict(row) for row in rows])
    return tables


def plan_summary(physical: object) -> dict[str, object]:
    """The observable shape of one physical plan: operator spine, leaf access paths, join choices."""
    document = describe_node(physical)
    spine: list[str] = []
    scans: dict[str, str] = {}
    joins: list[dict[str, object]] = []

    def walk(node: dict[str, object]) -> None:
        operator = str(node["operator"])
        spine.append(operator)
        if operator in ("full-scan", "index-scan", "index-lookup"):
            scans[str(node["table"])] = operator
        elif operator in ("hash-join", "index-nested-loop-join"):
            joins.append(
                {
                    "operator": operator,
                    "buildSide": node.get("buildSide"),
                    "outerSide": node.get("outerSide"),
                    "joinOrder": node.get("joinOrder"),
                    "leftKey": node.get("leftKey"),
                    "rightKey": node.get("rightKey"),
                }
            )
        for key in ("input", "left", "right"):
            child = node.get(key)
            if isinstance(child, dict):
                walk(child)

    walk(document)
    return {"spine": spine, "scans": scans, "joins": joins, "document": document}


def _public_summary(summary: dict[str, object]) -> dict[str, object]:
    return {key: summary[key] for key in ("spine", "scans", "joins")}


def case_dump(
    case: EquivalenceCase,
    summary_a: dict[str, object],
    summary_b: dict[str, object],
    result_a: object = None,
    result_b: object = None,
) -> str:
    """Everything needed to reproduce and locate a failure, in one JSON document."""
    payload: dict[str, object] = {
        "case": case.name,
        "sql": case.sql,
        "seeds": SEEDS,
        "tables": case.tables,
        "schemas": case.schemas,
        "variantA": case.variant_a.to_document(),
        "variantB": case.variant_b.to_document(),
        "planA": _public_summary(summary_a),
        "planB": _public_summary(summary_b),
    }
    if result_a is not None:
        payload["resultA"] = {"columns": result_a[0], "rows": result_a[1]}
    if result_b is not None:
        payload["resultB"] = {"columns": result_b[0], "rows": result_b[1]}
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def difference_problems(case: EquivalenceCase, summary_a: dict[str, object], summary_b: dict[str, object]) -> list[str]:
    """Why the two plans are *not* the intended different pair; empty means the case is valid."""
    problems: list[str] = []
    if summary_a["document"] == summary_b["document"]:
        problems.append("the two physical plans are identical, so the comparison would prove nothing")
    for dimension in case.expect:
        kind, _, argument = dimension.partition(":")
        if kind == "accessPath":
            first = summary_a["scans"].get(argument)
            second = summary_b["scans"].get(argument)
            if first == second:
                problems.append(f"access path for {argument!r} did not change ({first!r} in both plans)")
        elif kind == "buildSide":
            join_a = _only_join(summary_a, problems)
            join_b = _only_join(summary_b, problems)
            if join_a and join_b:
                if join_a["operator"] != "hash-join" or join_b["operator"] != "hash-join":
                    problems.append(f"expected two hash joins, got {join_a['operator']!r} and {join_b['operator']!r}")
                elif join_a["buildSide"] == join_b["buildSide"]:
                    problems.append(f"hash build side did not flip ({join_a['buildSide']!r} in both plans)")
        elif kind == "joinOperator":
            join_a = _only_join(summary_a, problems)
            join_b = _only_join(summary_b, problems)
            if join_a and join_b and join_a["operator"] == join_b["operator"]:
                problems.append(f"join operator did not change ({join_a['operator']!r} in both plans)")
        else:  # pragma: no cover - guards the suite itself
            problems.append(f"unknown expectation: {dimension!r}")
    return problems


def _only_join(summary: dict[str, object], problems: list[str]) -> dict[str, object] | None:
    joins = summary["joins"]
    if len(joins) != 1:
        problems.append(f"expected exactly one join in the plan, found {len(joins)}")
        return None
    return joins[0]


def assert_equivalent(test: unittest.TestCase, case: EquivalenceCase) -> None:
    """Plan the case's SQL through both catalogues, require different plans, require equal results."""
    statement = parse(case.sql)
    plan_a = plan(statement, _build_catalog(case, case.variant_a))
    plan_b = plan(statement, _build_catalog(case, case.variant_b))
    summary_a = plan_summary(plan_a.physical)
    summary_b = plan_summary(plan_b.physical)

    problems = difference_problems(case, summary_a, summary_b)
    if problems:
        test.fail(
            "case did not trigger the intended plan difference: "
            + "; ".join(problems)
            + "\n"
            + case_dump(case, summary_a, summary_b)
        )

    tables = _build_tables(case)
    result_a = execute(plan_a.physical, tables)
    result_b = execute(plan_b.physical, tables)

    columns_a, rows_a = result_a
    columns_b, rows_b = result_b
    test.assertEqual(
        columns_a,
        columns_b,
        msg="column names differ between the two physical plans\n" + case_dump(case, summary_a, summary_b, result_a, result_b),
    )
    # Row-by-row, in order: a set comparison would hide reordering and duplicate-row bugs.
    test.assertEqual(
        len(rows_a),
        len(rows_b),
        msg="row counts differ between the two physical plans\n" + case_dump(case, summary_a, summary_b, result_a, result_b),
    )
    for position, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        test.assertEqual(
            row_a,
            row_b,
            msg=f"row {position} differs between the two physical plans\n"
            + case_dump(case, summary_a, summary_b, result_a, result_b),
        )


# -- case tables --------------------------------------------------------------------------------------
def _access_path_cases() -> list[EquivalenceCase]:
    """Single table: an index declaration flips full-scan <-> index-scan downstream of everything."""
    tables = {"orders": ORDERS}
    plain = CatalogVariant()
    indexed = CatalogVariant(indexes={"orders": ("region",)})

    def make(name: str, sql: str) -> EquivalenceCase:
        return EquivalenceCase(name, sql, tables, plain, indexed, ("accessPath:orders",))

    return [
        make("filter", "SELECT id, amount FROM orders WHERE region = 'eu'"),
        make("and-not", "SELECT id, amount FROM orders WHERE region = 'eu' AND NOT (amount < 100.0)"),
        make("in-list", "SELECT id FROM orders WHERE region = 'eu' AND id IN (1, 2, 3, 4)"),
        make("star-is-null", "SELECT * FROM orders WHERE region = 'eu' AND note IS NULL"),
        make(
            "or-group-alias-order-limit",
            "SELECT region AS area, count(*) AS n, sum(amount) AS total FROM orders "
            "WHERE region = 'eu' AND (amount < 100.0 OR amount > 400.0) "
            "GROUP BY region ORDER BY region DESC LIMIT 3",
        ),
        make("filtered-empty", "SELECT id FROM orders WHERE region = 'zz'"),
        make(
            "aggregate-over-empty",
            "SELECT count(*) AS n, min(amount) AS lo, max(amount) AS hi FROM orders WHERE region = 'zz'",
        ),
    ]


def _build_side_cases() -> list[EquivalenceCase]:
    """Hash join: a stale distinct statistic flips the build side; results must not notice."""
    tables = {"orders": BUILD_ORDERS, "customers": BUILD_CUSTOMERS}
    honest = CatalogVariant()
    # Stale statistics: the planner is told region has 160 distinct values, so the pushed filter
    # looks selective enough to make orders the build side. Execution reads the same rows either way.
    stale = CatalogVariant(distinct={"orders": {"region": 160}})

    def make(name: str, sql: str) -> EquivalenceCase:
        return EquivalenceCase(name, sql, tables, honest, stale, ("buildSide",))

    join = "FROM orders INNER JOIN customers ON orders.cid = customers.cid WHERE orders.region = 'eu'"
    return [
        make("canonical-order", f"SELECT orders.id, customers.name {join}"),
        make("order-limit", f"SELECT orders.id, customers.name {join} ORDER BY orders.id DESC LIMIT 7"),
        make(
            "grouped-aggregate",
            "SELECT customers.name, count(*) AS n, sum(orders.amount) AS total "
            f"{join} GROUP BY customers.name ORDER BY customers.name",
        ),
    ]


def _join_operator_cases() -> list[EquivalenceCase]:
    """An index on the join key flips hash-join <-> index-nested-loop-join, in both outer directions."""
    tables = {"customers": JOIN_CUSTOMERS, "orders": JOIN_ORDERS}
    plain = CatalogVariant()
    indexed = CatalogVariant(indexes={"orders": ("cid",)})

    def make(name: str, sql: str, case_tables=None, schemas=None) -> EquivalenceCase:
        return EquivalenceCase(
            name, sql, case_tables or tables, plain, indexed, ("joinOperator",), schemas or {}
        )

    outer_left = "FROM customers INNER JOIN orders ON customers.cid = orders.cid"
    outer_right = "FROM orders INNER JOIN customers ON customers.cid = orders.cid"
    return [
        make("outer-left-canonical-order", f"SELECT customers.name, orders.id, orders.amount {outer_left}"),
        make(
            "pushed-predicates-both-sides",
            f"SELECT customers.name, orders.id, orders.amount {outer_left} "
            "WHERE orders.amount > 100.0 AND customers.name <> 'x'",
        ),
        make("outer-right", f"SELECT orders.id, customers.name {outer_right}"),
        make("star-limit", f"SELECT * {outer_left} LIMIT 9"),
        make(
            "empty-left-table",
            f"SELECT customers.name, orders.id {outer_left}",
            case_tables={"customers": [], "orders": ORDERS},
            schemas={"customers": CUSTOMER_SCHEMA},
        ),
        make(
            "empty-right-table",
            f"SELECT orders.id, customers.name {outer_right}",
            case_tables={"orders": ORDERS, "customers": []},
            schemas={"customers": CUSTOMER_SCHEMA},
        ),
        make("filtered-empty-side", f"SELECT customers.name, orders.id {outer_left} WHERE customers.name = 'zz'"),
    ]


ACCESS_PATH_CASES = _access_path_cases()
BUILD_SIDE_CASES = _build_side_cases()
JOIN_OPERATOR_CASES = _join_operator_cases()


# -- the datasets themselves ---------------------------------------------------------------------------
class DatasetTests(unittest.TestCase):
    """The generators are deterministic and really produce the edge shapes the cases rely on."""

    def test_generators_reproduce_from_their_seeds(self) -> None:
        self.assertEqual(generate_orders(ORDERS_SEED, 60), ORDERS)
        self.assertEqual(generate_orders(BUILD_LEFT_SEED, 80), BUILD_ORDERS)
        self.assertEqual(generate_customers(BUILD_RIGHT_SEED, 20), BUILD_CUSTOMERS)
        self.assertEqual(generate_orders(JOIN_ORDERS_SEED, 200), JOIN_ORDERS)

    def test_only_documented_json_scalars_appear(self) -> None:
        for rows in (ORDERS, BUILD_ORDERS, BUILD_CUSTOMERS, JOIN_ORDERS, JOIN_CUSTOMERS):
            for row in rows:
                for value in row.values():
                    self.assertIsInstance(value, (str, int, float, type(None)))
                    self.assertNotIsInstance(value, bool)

    def test_duplicate_and_null_join_keys_are_present(self) -> None:
        for rows in (ORDERS, BUILD_ORDERS, JOIN_ORDERS):
            keys = [row["cid"] for row in rows]
            self.assertIn(None, keys)
            self.assertLess(len({key for key in keys if key is not None}), len(rows))

    def test_missing_fields_and_null_values_are_present(self) -> None:
        self.assertTrue(any("note" not in row for row in ORDERS))
        self.assertTrue(any(row.get("note") is None for row in ORDERS))
        self.assertTrue(any("name" not in row for row in BUILD_CUSTOMERS))
        self.assertTrue(any("name" not in row for row in JOIN_CUSTOMERS))


# -- equivalence per plan dimension ---------------------------------------------------------------------
class AccessPathEquivalenceTests(unittest.TestCase):
    """full-scan vs index-scan (lever: index declaration), one subTest per query shape."""

    def test_all_access_path_cases(self) -> None:
        for case in ACCESS_PATH_CASES:
            with self.subTest(case=case.name):
                assert_equivalent(self, case)


class HashBuildSideEquivalenceTests(unittest.TestCase):
    """hash-join building left vs right (lever: distinct statistics)."""

    def test_all_build_side_cases(self) -> None:
        for case in BUILD_SIDE_CASES:
            with self.subTest(case=case.name):
                assert_equivalent(self, case)


class JoinOperatorEquivalenceTests(unittest.TestCase):
    """hash-join vs index-nested-loop-join (lever: index declaration), both outer directions."""

    def test_all_join_operator_cases(self) -> None:
        for case in JOIN_OPERATOR_CASES:
            with self.subTest(case=case.name):
                assert_equivalent(self, case)


# -- the public reconcile entry point ---------------------------------------------------------------------
def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ReconcileEquivalenceTests(unittest.TestCase):
    """`reconcile` on the index-driven cases: identical=true, exit 0, and plan summaries that
    actually show the two different choices. (Build-side flips need stale statistics and empty
    tables need a declared schema, neither of which a JSONL file can express, so those stay at
    library level above.)
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.orders = self._write("orders.jsonl", ORDERS)
        self.join_orders = self._write("join_orders.jsonl", JOIN_ORDERS)
        self.join_customers = self._write("join_customers.jsonl", JOIN_CUSTOMERS)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, name: str, rows: list[dict[str, object]]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def _join_bindings(self) -> list[str]:
        return ["--table", f"customers={self.join_customers}", "--table", f"orders={self.join_orders}"]

    def test_single_table_access_paths_reconcile(self) -> None:
        sql = "SELECT id, amount FROM orders WHERE region = 'eu' AND NOT (amount < 100.0)"
        code, out, err = run_cli(["reconcile", "--sql", sql, "--table", self.orders, "--index", "region"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        first, second = document["plans"]
        self.assertEqual(first["accessPath"], "index-scan")
        self.assertEqual(second["accessPath"], "full-scan")
        self.assertNotEqual(first, second)

    def test_filtered_empty_result_reconciles_with_zero_rows(self) -> None:
        code, out, err = run_cli(
            ["reconcile", "--sql", "SELECT id FROM orders WHERE region = 'zz'", "--table", self.orders, "--index", "region"]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["rows"], 0)
        self.assertNotEqual(document["plans"][0]["accessPath"], document["plans"][1]["accessPath"])

    def test_join_operators_reconcile_outer_left(self) -> None:
        sql = "SELECT customers.name, orders.id FROM customers INNER JOIN orders ON customers.cid = orders.cid"
        code, out, err = run_cli(["reconcile", "--sql", sql, *self._join_bindings(), "--index", "orders.cid"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        first, second = document["plans"]
        self.assertEqual(first["joinOperator"], "index-nested-loop-join")
        self.assertEqual(second["joinOperator"], "hash-join")
        self.assertIn("joinOrder", first)
        self.assertIn("joinOrder", second)

    def test_join_operators_reconcile_outer_right(self) -> None:
        sql = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON customers.cid = orders.cid"
        code, out, err = run_cli(["reconcile", "--sql", sql, *self._join_bindings(), "--index", "orders.cid"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        first, second = document["plans"]
        self.assertEqual(first["joinOperator"], "index-nested-loop-join")
        self.assertEqual(first["joinOrder"], ["customers", "orders"])
        self.assertEqual(second["joinOperator"], "hash-join")

    def test_star_and_limit_reconcile(self) -> None:
        sql = "SELECT * FROM customers INNER JOIN orders ON customers.cid = orders.cid LIMIT 9"
        code, out, err = run_cli(["reconcile", "--sql", sql, *self._join_bindings(), "--index", "orders.cid"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["rows"], 9)
        self.assertNotEqual(document["plans"][0]["joinOperator"], document["plans"][1]["joinOperator"])

    def test_reconcile_output_file_is_written_atomically(self) -> None:
        target = os.path.join(self.directory.name, "report.json")
        sql = "SELECT id, amount FROM orders WHERE region = 'eu'"
        code, out, err = run_cli(
            ["reconcile", "--sql", sql, "--table", self.orders, "--index", "region", "--output", target]
        )
        self.assertEqual((code, out, err), (EXIT_OK, "", ""))
        with open(target, encoding="utf-8") as handle:
            document = json.loads(handle.read())
        self.assertTrue(document["identical"])
        self.assertEqual(document["plans"][0]["accessPath"], "index-scan")
        self.assertEqual(document["plans"][1]["accessPath"], "full-scan")


if __name__ == "__main__":
    unittest.main()
