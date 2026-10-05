"""Deterministic physical-plan equivalence tests.

The promise under test: one legal statement, executed through two *genuinely different* physical
plans, must return the same columns and the same rows in the same order -- with or without
``ORDER BY``.  Every case is built from fixed-seed data (``SEED`` below) so a failure reproduces
bit-for-bit, and every case first proves the two plans it compares really differ (access path,
hash build side, or join operator).  A case whose data fails to trigger its target plan fails
loudly instead of silently degrading to a plan compared with itself.

Three physical choices are exercised against the public ``reconcile`` command (exit code 0 and
``identical: true`` are part of the contract):

  * single table ......... full-scan  vs  index-scan          (an index on the filtered column)
  * INNER JOIN ........... hash-join  vs  index-nested-loop   (an index on the inner join key,
    both outer directions)

The hash-join build side cannot be moved by an ``--index`` declaration alone -- building left or
right is decided from the *filtered* row estimates -- so those cases plan the same statement
twice in-process over identical rows with two distinct-count profiles (exactly the "data scale and
distinct statistics trigger either build side" knob the README documents), execute both, and add
a third plan built from the table-derived statistics as a non-degeneracy check.

Rows below a join are compared as ordered lists, never as sets: without ``ORDER BY`` the contract
is logical-left original order then logical-right original order, duplicate join keys included.

Only documented JSON scalars appear in the data (null, integers, decimals, strings); every query
uses documented syntax.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import tempfile
import unittest

from sqlplan.cli import main
from sqlplan.executor import Table, execute
from sqlplan.parser import parse
from sqlplan.planner import (
    Catalog,
    HashJoin,
    IndexNestedLoopJoin,
    TableInfo,
    describe_node,
    plan,
)

SEED = 20261005


# -------------------------------------------------------------------------------------------------
# Fixed-seed fixtures. Changing SEED changes every generated row; every expected value in the
# oracle assertions below was derived from this exact seed.
# -------------------------------------------------------------------------------------------------
def _fixtures() -> dict[str, dict[str, list[dict[str, object]]]]:
    rng = random.Random(SEED)

    # 60 rows: region is null every seventh row (8 nulls), two dense values otherwise so an index
    # on region pays off; `note` is absent on every fifth row (missing field) and null on some
    # others; `amount` mixes integer-valued decimals with fractions.
    events: list[dict[str, object]] = []
    for i in range(1, 61):
        m = i % 7
        row: dict[str, object] = {
            "id": i,
            "seq": i,
            "region": None if m == 0 else ("eu" if m in (1, 2) else "us"),
            "amount": round(10 + i * 1.25 + rng.random(), 2),
        }
        if i % 5:  # rows 5,10,...,60 deliberately omit `note`
            row["note"] = rng.choice(["x", "y", None])
        events.append(row)

    # 40/40 join pair with repeated join keys (pattern 1,1,2,2,3,NULL) and sparse side columns:
    # `a`/`b` are pinned by pushed equality filters; `name` is missing from two customers rows.
    pattern = (1, 1, 2, 2, 3, None)
    orders = [
        {"id": i, "seq": i, "cid": pattern[i % 6], "a": ((i * 7 + 3) % 20) + 1, "g": f"g{i % 4}"}
        for i in range(1, 41)
    ]
    customers = [
        {"cid": pattern[i % 6], "seq": i, "b": ((i * 11 + 5) % 20) + 1,
         "name": (None if i % 8 == 0 else f"n{i}"), "ref": i % 4}
        for i in range(1, 41)
    ]
    for i in (9, 10):  # missing field, not null
        customers[i - 1].pop("name")

    # Same shape, but dense on the side-filter columns so the two statistic profiles below can
    # pull the hash build estimate to either side over these identical rows.
    flip_orders = [
        {"id": i, "seq": i, "cid": pattern[i % 6], "a": 7 if i % 2 else 8,
         "g": f"g{i % 4}", "note": ("z" if i % 3 else None)}
        for i in range(1, 41)
    ]
    flip_customers = [
        {"cid": pattern[i % 6], "seq": i, "b": 9 if i % 2 else 10,
         "name": (None if i in (9, 10) else f"n{i}"), "ref": i % 4}
        for i in range(1, 41)
    ]

    # Small left outer (4 rows: duplicate key 1, key 2, a null key) vs big right (60 rows,
    # 30 distinct join keys) with an index on the right join key -> left-outer index nested loop.
    small_orders = [
        {"id": 1, "seq": 1, "k": 1, "amount": 10.0, "note": "p"},
        {"id": 2, "seq": 2, "k": 1, "amount": 20.0, "note": "q"},
        {"id": 3, "seq": 3, "k": 2, "amount": None, "note": None},
        {"id": 4, "seq": 4, "k": None, "amount": 40.0, "note": None},
    ]
    big_customers = [
        {"cid": ((i - 1) % 30) + 1, "seq": i, "v": float(i),
         "tier": ("bronze", "silver", "gold")[i % 3]}
        for i in range(1, 61)
    ]
    for i in (5, 17):  # sparse optional field
        big_customers[i - 1]["name"] = f"c{i}"

    # Mirror image: big left (60 rows) with an indexed join key vs small right outer (4 rows,
    # duplicate key 1 and a null key) -> right-outer index nested loop.
    big_orders = [
        {"id": i, "seq": i, "cid": ((i - 1) % 30) + 1, "amount": float(i),
         "g": ("x" if i % 3 else None)}
        for i in range(1, 61)
    ]
    small_customers = [
        {"k": 1, "seq": 1, "who": "a"},
        {"k": 1, "seq": 2, "who": "a2"},
        {"k": 2, "seq": 3, "who": "b"},
        {"k": None, "seq": 4, "who": "z"},
    ]

    return {
        "events": {"events": events},
        "pairs": {"orders": orders, "customers": customers},
        "flip": {"orders": flip_orders, "customers": flip_customers},
        "small_big": {"orders": small_orders, "customers": big_customers},
        "big_small": {"orders": big_orders, "customers": small_customers},
    }


DATA = _fixtures()

# Indexes declared per reconcile scenario (qualified for joins, bare for one table).
INDEXES: dict[str, list[str]] = {
    "events": ["region"],
    "pairs": ["customers.cid"],
    "small_big": ["customers.cid"],
    "big_small": ["orders.cid"],
}


# -------------------------------------------------------------------------------------------------
# Harness
# -------------------------------------------------------------------------------------------------
def _columns_of(rows: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({column for row in rows for column in row}))


def _distinct(rows: list[dict[str, object]], column: str) -> int:
    return len({row.get(column) for row in rows})


def _join_node(plan_result) -> object:
    node = plan_result.physical
    while not isinstance(node, (HashJoin, IndexNestedLoopJoin)):
        node = node.input
    return node


def _plan_summary(node: object) -> dict[str, object]:
    """Compact, comparable summary plus the full physical document for diagnostics."""
    document = describe_node(node)
    summary: dict[str, object] = {
        "operator": document["operator"],
        "physical": document,
    }
    if isinstance(node, HashJoin):
        summary["buildSide"] = node.build_side
        summary["joinOrder"] = document["joinOrder"]
    if isinstance(node, IndexNestedLoopJoin):
        summary["outerSide"] = node.outer_side
        summary["joinOrder"] = document["joinOrder"]
    return summary


def _render_report(
    sql: str,
    named: dict[str, list[dict[str, object]]],
    indexes: list[str],
    summary_a: object,
    summary_b: object,
    result_a: object,
    result_b: object,
    extra: str = "",
) -> str:
    """A failure report that names everything needed to reproduce one divergence."""
    parts = [
        "",
        "=" * 78,
        "physical plans disagreed (or failed to diverge)",
        f"SQL: {sql}",
        f"index declarations: {json.dumps(indexes, ensure_ascii=False)}",
    ]
    for name, rows in named.items():
        parts.append(f"table {name} ({len(rows)} rows):")
        parts.append(json.dumps(rows, ensure_ascii=False))
    parts.extend([
        "plan A summary:",
        json.dumps(summary_a, ensure_ascii=False, sort_keys=True, indent=1),
        "plan B summary:",
        json.dumps(summary_b, ensure_ascii=False, sort_keys=True, indent=1),
        "result A (columns, rows):",
        json.dumps(result_a, ensure_ascii=False, sort_keys=True, indent=1),
        "result B (columns, rows):",
        json.dumps(result_b, ensure_ascii=False, sort_keys=True, indent=1),
    ])
    if extra:
        parts.extend(["detail:", extra])
    return "\n".join(parts)


class EquivalenceCase(unittest.TestCase):
    """Shared harness: JSONL fixtures on disk, the real CLI, and in-process dual execution."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, name: str, rows: list[dict[str, object]]) -> str:
        path = os.path.join(self.directory.name, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _argv(self, command: str, key: str, sql: str, *, with_index: bool) -> list[str]:
        named = DATA[key]
        argv = [command, "--sql", sql]
        if len(named) == 1:
            argv += ["--table", self._write(f"{key}_t", next(iter(named.values())))]
        else:
            for name, rows in named.items():
                argv += ["--table", f"{name}={self._write(f'{key}_{name}', rows)}"]
        if with_index:
            argv += [f"--index={value}" for value in INDEXES[key]]
        return argv

    def _cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def reconcile(self, key: str, sql: str) -> tuple[int, dict[str, object], str]:
        code, out, err = self._cli(self._argv("reconcile", key, sql, with_index=True))
        try:
            return code, json.loads(out), err
        except json.JSONDecodeError:
            return code, {"_unparsed_stdout": out}, err

    def run_sql(self, key: str, sql: str) -> tuple[int, dict[str, object], str]:
        code, out, err = self._cli(self._argv("run", key, sql, with_index=False))
        return code, (json.loads(out) if out else {}), err

    def execute_inproc(
        self,
        key: str,
        sql: str,
        *,
        with_index: bool,
        stats_overrides: dict[str, dict[str, int]] | None = None,
    ):
        """Plan and execute one statement against one catalogue view of the fixed rows."""
        catalog = Catalog()
        tables: dict[str, Table] = {}
        for name, rows in DATA[key].items():
            columns = _columns_of(rows)
            distinct = {column: _distinct(rows, column) for column in columns}
            distinct.update((stats_overrides or {}).get(name, {}))
            indexes = tuple(INDEXES.get(key, [])) if with_index else ()
            # join index specs are qualified; a single-table spec is bare.
            if len(DATA[key]) > 1:
                indexes = tuple(spec.split(".", 1)[1] for spec in indexes if spec.startswith(f"{name}."))
            catalog.add(TableInfo(name, columns, len(rows), distinct, indexes))
            tables[name] = Table(name, columns, [dict(row) for row in rows])
        result = plan(parse(sql), catalog)
        return result, execute(result.physical, tables)

    def assert_reconcile_equivalent(
        self,
        key: str,
        sql: str,
        *,
        expect_indexed: tuple[str, str | None],
        expect_baseline: tuple[str, str | None],
    ) -> tuple[dict[str, object], tuple[list[str], list[dict[str, object]]], tuple[list[str], list[dict[str, object]]]]:
        """The public guarantee: reconcile exits 0 with identical=true, over two unlike plans.

        Expect pairs are (accessPath, joinOperator) for each catalogue in reconcile order
        (with-index first). Asserting both proves the compared choices really are the ones the
        case was designed to trigger.
        """
        code, report, err = self.reconcile(key, sql)
        if code != 0 or err:
            self.fail(_render_report(sql, DATA[key], INDEXES[key], report, None, None, None,
                                    extra=f"reconcile exit={code} stderr={err}"))
        plans = report["plans"]
        indexed, baseline = plans[0], plans[1]
        signature_a = (indexed["accessPath"], indexed["joinOperator"])
        signature_b = (baseline["accessPath"], baseline["joinOperator"])
        chosen_a, result_a = self.execute_inproc(key, sql, with_index=True)
        chosen_b, result_b = self.execute_inproc(key, sql, with_index=False)
        # the reconcile entry plus the full physical operator tree, so a failure points at the
        # exact operator whose choice diverged (scan leaf, join node, build side, order)
        summary_a = {"entry": indexed, "physical": describe_node(chosen_a.physical)}
        summary_b = {"entry": baseline, "physical": describe_node(chosen_b.physical)}
        if signature_a != expect_indexed or signature_b != expect_baseline or signature_a == signature_b:
            self.fail(_render_report(
                sql, DATA[key], INDEXES[key], summary_a, summary_b, result_a, result_b,
                extra=f"expected {expect_indexed} vs {expect_baseline}, got {signature_a} vs {signature_b}",
            ))
        self.assertTrue(report["identical"])
        # Explicit, ordered, duplicate-sensitive re-check: never compare these as sets.
        self.assertEqual(
            result_a, result_b,
            msg=_render_report(sql, DATA[key], INDEXES[key], summary_a, summary_b, result_a, result_b),
        )
        return report, result_a, result_b


# -------------------------------------------------------------------------------------------------
# Single table: full-scan vs index-scan
# -------------------------------------------------------------------------------------------------
class SingleTableAccessPathTests(EquivalenceCase):
    KEY = "events"

    def test_s1_projection_aliases_and_equality_index_scan(self) -> None:
        sql = "SELECT id AS k, amount AS v FROM events WHERE region = 'eu'"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        # no ORDER BY: original table order, independent plain-Python oracle
        expected = [
            {"k": row["id"], "v": row["amount"]}
            for row in DATA["events"]["events"] if row["region"] == "eu"
        ]
        self.assertEqual(result_a[1], expected)
        self.assertEqual(result_b[1], expected)
        self.assertEqual(result_a[0], ["k", "v"])

    def test_s2_and_or_not_in_isnull_predicate_soup_keeps_table_order(self) -> None:
        sql = (
            "SELECT id, note FROM events WHERE region = 'eu' "
            "AND (amount > 30 OR note IS NULL) AND NOT (amount > 70) "
            "AND id IN (2, 8, 15, 16, 22, 23)"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        # four rows survive, in original order; missing/sparse notes come through as null
        self.assertEqual(
            result_a[1],
            [{"id": 15, "note": None}, {"id": 16, "note": None},
             {"id": 22, "note": None}, {"id": 23, "note": "x"}],
        )

    def test_s3_star_with_missing_fields_and_null_region(self) -> None:
        sql = "SELECT * FROM events WHERE region = 'us'"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        rows = DATA["events"]["events"]
        columns = list(_columns_of(rows))
        self.assertEqual(result_a[0], columns)  # star -> every column, sorted scan order
        expected = [{column: row.get(column) for column in columns}
                    for row in rows if row["region"] == "us"]
        self.assertEqual(result_a[1], expected)  # missing note key materialises as null

    def test_s4_grouped_aggregates_over_indexed_and_full_scan(self) -> None:
        sql = (
            "SELECT note, count(*) AS c, sum(amount) AS s FROM events "
            "WHERE region = 'eu' GROUP BY note ORDER BY note"
        )
        _, result_a, _ = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(result_a[0], ["note", "c", "s"])
        self.assertEqual([(row["note"], row["c"]) for row in result_a[1]],
                         [("x", 7), ("y", 4), (None, 7)])
        self.assertAlmostEqual(sum(row["s"] for row in result_a[1]),
                               sum(row["amount"] for row in DATA["events"]["events"]
                                   if row["region"] == "eu"))

    def test_s5_order_by_desc_and_limit_above_different_scans(self) -> None:
        sql = "SELECT id, amount FROM events WHERE region = 'eu' ORDER BY amount DESC LIMIT 5"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(
            [(row["id"], row["amount"]) for row in result_a[1]],
            [(58, 83.2), (57, 81.65), (51, 73.86), (50, 72.54), (44, 65.25)],
        )
        self.assertEqual(result_a[1], result_b[1])

    def test_s6_filtered_to_empty_set_is_equivalent_and_ordered_empty(self) -> None:
        sql = "SELECT id FROM events WHERE region = 'eu' AND id > 1000"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(result_a, (["id"], []))
        self.assertEqual(result_b, (["id"], []))

    def test_s7_global_aggregate_over_empty_set_emits_one_null_row(self) -> None:
        sql = (
            "SELECT count(*) AS c, sum(amount) AS s, avg(amount) AS a, min(amount) AS lo "
            "FROM events WHERE region = 'zzz'"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(result_a, (["c", "s", "a", "lo"],
                                    [{"c": 0, "s": None, "a": None, "lo": None}]))
        self.assertEqual(result_a, result_b)

    def test_s8_limit_zero_is_equivalent_and_ordered_empty(self) -> None:
        sql = "SELECT id FROM events WHERE region = 'us' LIMIT 0"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(result_a, (["id"], []))
        self.assertEqual(result_b, (["id"], []))

    def test_s9_in_list_with_order_by(self) -> None:
        sql = "SELECT id AS i, note FROM events WHERE region = 'us' AND id IN (5, 10, 15) ORDER BY id"
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        self.assertEqual(result_a[1], [{"i": 5, "note": None}, {"i": 10, "note": None}])
        self.assertEqual(result_a[1], result_b[1])

    def test_s10_not_in_and_is_not_null_without_order_by_keeps_table_order(self) -> None:
        sql = (
            "SELECT id FROM events WHERE region = 'us' "
            "AND id NOT IN (5, 10, 15) AND note IS NOT NULL"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            self.KEY, sql, expect_indexed=("index-scan", None), expect_baseline=("full-scan", None))
        expected_ids = [3, 6, 11, 12, 13, 18, 26, 32, 38, 39, 41, 47, 52, 53, 54, 59]
        self.assertEqual([row["id"] for row in result_a[1]], expected_ids)
        self.assertEqual(result_a[1], result_b[1])


# -------------------------------------------------------------------------------------------------
# INNER JOIN: hash-join vs index-nested-loop-join (both outer directions)
# -------------------------------------------------------------------------------------------------
class JoinIndexNestedLoopTests(EquivalenceCase):
    def test_j1_pushed_predicates_on_both_sides_flip_hash_to_inlj(self) -> None:
        sql = (
            "SELECT orders.id AS oid, customers.name AS cname FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9"
        )
        report, _, _ = self.assert_reconcile_equivalent(
            "pairs", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(report["rows"], 1)
        code, document, err = self.run_sql("pairs", sql)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(document["rows"], [{"oid": 9, "cname": "n33"}])

    def test_j6_cross_of_two_filters_is_empty_under_both_join_operators(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9 AND orders.g = 'zz'"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "pairs", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(result_a, (["id"], []))
        self.assertEqual(result_b, (["id"], []))

    # -- small left outer, big indexed right ----------------------------------------------------
    def test_jb1_in_and_range_pushdown_left_outer_inlj(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.k = customers.cid "
            "WHERE orders.k IN (1, 2) AND customers.v >= 31"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        pairs = [(row["orders.seq"], row["customers.seq"]) for row in result_a[1]]
        self.assertEqual(pairs, [(1, 31), (2, 31), (3, 32)])  # ordered, duplicates kept
        self.assertEqual(result_a[1], result_b[1])

    def test_jb2_star_order_by_and_limit_agree(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.k = customers.cid "
            "ORDER BY orders.id, customers.cid LIMIT 5"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        pairs = [(row["orders.id"], row["customers.cid"]) for row in result_a[1]]
        self.assertEqual(pairs, [(1, 1), (1, 1), (2, 1), (2, 1), (3, 2)])
        self.assertEqual(result_a[0], result_b[0])

    def test_jb3_grouped_aggregates_with_null_measure_ignored(self) -> None:
        sql = (
            "SELECT customers.tier AS tier, count(*) AS n, sum(orders.amount) AS total "
            "FROM orders INNER JOIN customers ON orders.k = customers.cid "
            "GROUP BY customers.tier ORDER BY customers.tier"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(
            [(row["tier"], row["n"], row["total"]) for row in result_a[1]],
            # bronze joins nothing; gold's two matches both carry the NULL amount, so its SUM is
            # NULL (every non-count aggregate over an all-NULL group is NULL), silver sums 20+40
            [("gold", 2, None), ("silver", 4, 60.0)],
        )

    def test_jb4_filtered_empty_inner_lookup(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.k = customers.cid "
            "WHERE customers.v > 1000"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(result_a, (["id"], []))
        self.assertEqual(result_b, (["id"], []))

    def test_jb5_or_isnull_and_not_soup_around_inlj(self) -> None:
        sql = (
            "SELECT orders.id AS oid, customers.cid AS cid, orders.note FROM orders "
            "INNER JOIN customers ON orders.k = customers.cid "
            "WHERE (customers.v < 15 OR orders.amount IS NULL) AND NOT (customers.tier = 'gold')"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        # outer order: order 1 then 2, the order-3 row only enters via the IS NULL arm
        self.assertEqual(
            result_a[1],
            [{"oid": 1, "cid": 1, "note": "p"}, {"oid": 2, "cid": 1, "note": "q"}],
        )
        self.assertEqual(result_a[1], result_b[1])

    def test_jb6_global_aggregates_including_null_join_key_excluded(self) -> None:
        sql = (
            "SELECT count(*) AS n, sum(orders.amount) AS s, min(customers.v) AS lo, "
            "max(customers.v) AS hi FROM orders INNER JOIN customers ON orders.k = customers.cid"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "small_big", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(result_a[1], [{"n": 6, "s": 60.0, "lo": 1.0, "hi": 32.0}])
        self.assertEqual(result_a[1], result_b[1])

    # -- big indexed left, small right outer ----------------------------------------------------
    def test_jc1_residual_cross_side_filter_with_order_limit_right_outer_inlj(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.k "
            "WHERE customers.k IS NOT NULL AND orders.id <> customers.k "
            "ORDER BY orders.id, customers.seq LIMIT 4"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "big_small", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        # residual inequality drops the self-matches (1->k1, 2->k2); limit 4 keeps 3 survivors
        self.assertEqual([row["orders.id"] for row in result_a[1]], [31, 31, 32])
        self.assertEqual(result_a[1], result_b[1])

    def test_jc2_duplicate_right_keys_multiply_with_star_right_outer(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.k "
            "WHERE customers.k = 1"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "big_small", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        pairs = [(row["orders.seq"], row["customers.seq"]) for row in result_a[1]]
        self.assertEqual(pairs, [(1, 1), (1, 2), (31, 1), (31, 2)])  # full 2x2 product
        self.assertEqual(result_a[1], result_b[1])

    def test_jc3_global_min_max_sum_count_right_outer(self) -> None:
        sql = (
            "SELECT count(*) AS n, min(orders.amount) AS lo, max(orders.amount) AS hi "
            "FROM orders INNER JOIN customers ON orders.cid = customers.k"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "big_small", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(result_a[1], [{"n": 6, "lo": 1.0, "hi": 32.0}])
        self.assertEqual(result_a[1], result_b[1])

    def test_jc5_right_outer_lookup_matches_nothing(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = customers.k "
            "WHERE customers.k = 99"
        )
        _, result_a, result_b = self.assert_reconcile_equivalent(
            "big_small", sql,
            expect_indexed=("full-scan", "index-nested-loop-join"),
            expect_baseline=("full-scan", "hash-join"),
        )
        self.assertEqual(result_a, (["id"], []))
        self.assertEqual(result_b, (["id"], []))


# -------------------------------------------------------------------------------------------------
# Hash build side: the same statement planned against two distinct-count profiles. These run
# in-process because reconcile's knob is index presence, which cannot move a hash build side:
# build choice is a pure function of the *filtered* row estimates. A third plan over the
# table-derived statistics is executed as well and must agree, proving the flip itself is benign.
# -------------------------------------------------------------------------------------------------
class HashBuildSideTests(EquivalenceCase):
    KEY = "flip"
    # Profile A makes the left side's pinned column look dense (left survives ~1/20 of its rows),
    # profile B mirrors it; the filtered-row estimates then swap the cheaper build side.
    PROFILE_A = {"orders": {"a": 20}, "customers": {"b": 2}}
    PROFILE_B = {"orders": {"a": 2}, "customers": {"b": 20}}

    def _both_plans(self, sql: str):
        plan_a, result_a = self.execute_inproc(
            self.KEY, sql, with_index=False, stats_overrides=self.PROFILE_A)
        plan_b, result_b = self.execute_inproc(
            self.KEY, sql, with_index=False, stats_overrides=self.PROFILE_B)
        plan_c, result_c = self.execute_inproc(self.KEY, sql, with_index=False)
        join_a, join_b, join_c = _join_node(plan_a), _join_node(plan_b), _join_node(plan_c)
        for label, join in (("A", join_a), ("B", join_b), ("C", join_c)):
            if not isinstance(join, HashJoin):
                self.fail(_render_report(
                    sql, DATA[self.KEY], [],
                    _plan_summary(join_a), _plan_summary(join_b), result_a, result_b,
                    extra=f"profile {label} failed to produce a hash-join, got {type(join).__name__}",
                ))
        if join_a.build_side == join_b.build_side:
            self.fail(_render_report(
                sql, DATA[self.KEY], [],
                _plan_summary(join_a), _plan_summary(join_b), result_a, result_b,
                extra="both distinct profiles built the same side -- case would be a self-comparison",
            ))
        return ((plan_a, join_a, result_a), (plan_b, join_b, result_b), (plan_c, join_c, result_c))

    def _assert_three_agree(self, sql: str) -> tuple:
        (_, join_a, result_a), (_, join_b, result_b), (_, _, result_c) = self._both_plans(sql)
        self.assertEqual(join_a.build_side, "left")
        self.assertEqual(join_b.build_side, "right")
        # joinOrder publishes the build-first ordering, so the divergence is visible in the summary
        self.assertEqual(describe_node(join_a)["joinOrder"], ["orders", "customers"])
        self.assertEqual(describe_node(join_b)["joinOrder"], ["customers", "orders"])
        self.assertEqual(
            result_a, result_b,
            msg=_render_report(sql, DATA[self.KEY], [],
                               _plan_summary(join_a), _plan_summary(join_b), result_a, result_b),
        )
        self.assertEqual(result_a, result_c)
        return result_a

    def test_j2_star_duplicate_keys_nulls_missing_fields_both_build_sides(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9"
        )
        result = self._assert_three_agree(sql)
        orders, customers = DATA[self.KEY]["orders"], DATA[self.KEY]["customers"]
        reference = sorted(
            (left["seq"], right["seq"])
            for left in orders
            for right in customers
            if left["cid"] is not None and left["cid"] == right["cid"]
            and left["a"] == 7 and right["b"] == 9
        )
        pairs = [(row["orders.seq"], row["customers.seq"]) for row in result[1]]
        self.assertEqual(len(pairs), 98)
        self.assertEqual(pairs, reference)  # list equality: exact order and duplicate pairs
        self.assertTrue(all(row["orders.cid"] is not None for row in result[1]))  # nulls never match
        self.assertTrue(any("customers.name" in row and row["customers.name"] is None
                            for row in result[1]))  # missing field surfaces as null

    def test_j3_grouped_aggregates_both_build_sides(self) -> None:
        sql = (
            "SELECT customers.name AS who, count(*) AS n, sum(orders.id) AS s FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9 GROUP BY customers.name ORDER BY customers.name"
        )
        result = self._assert_three_agree(sql)
        # independent oracle grouped on the right table's surviving rows
        orders = [row for row in DATA[self.KEY]["orders"] if row["a"] == 7]
        customers = [row for row in DATA[self.KEY]["customers"] if row["b"] == 9]
        groups: dict[object, dict[str, object]] = {}
        for right in customers:
            matched = [left["id"] for left in orders
                       if left["cid"] is not None and left["cid"] == right["cid"]]
            if matched:
                groups[right.get("name")] = {"who": right.get("name"), "n": len(matched),
                                             "s": float(sum(matched))}
        expected = [groups[key] for key in sorted(groups, key=lambda v: (v is None, str(v)))]
        self.assertEqual(result[1], expected)

    def test_j5_order_by_and_limit_both_build_sides(self) -> None:
        sql = (
            "SELECT orders.id AS oid, customers.seq AS cs FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9 "
            "ORDER BY orders.id, customers.seq LIMIT 6"
        )
        result = self._assert_three_agree(sql)
        self.assertEqual(
            result[1],
            [{"oid": 1, "cs": 1}, {"oid": 1, "cs": 7}, {"oid": 1, "cs": 13},
             {"oid": 1, "cs": 19}, {"oid": 1, "cs": 25}, {"oid": 1, "cs": 31}],
        )

    def test_ja4_or_not_and_residual_cross_side_filter_both_build_sides(self) -> None:
        sql = (
            "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.a = 7 AND customers.b = 9 "
            "AND (orders.g = 'g1' OR customers.name IS NULL) "
            "AND orders.id <> customers.ref"
        )
        result = self._assert_three_agree(sql)
        orders, customers = DATA[self.KEY]["orders"], DATA[self.KEY]["customers"]
        reference = sorted(
            (left["seq"], right["seq"])
            for left in orders
            for right in customers
            if left["cid"] is not None and left["cid"] == right["cid"]
            and left["a"] == 7 and right["b"] == 9
            and (left["g"] == "g1" or right.get("name") is None)
            and left["id"] != right["ref"]
        )
        pairs = [(row["orders.seq"], row["customers.seq"]) for row in result[1]]
        self.assertEqual(len(pairs), 49)
        self.assertEqual(pairs, reference)  # pushed single-side + residual above-join both agree


# -------------------------------------------------------------------------------------------------
# Empty input. A CLI-loaded empty table has no columns (statistics derive from rows), so the empty
# case is planned in-process with an explicit schema, exactly as the join unit tests do. The index
# sits on the NON-empty side: its empty partner then drives zero lookups as the INLJ outer.
# -------------------------------------------------------------------------------------------------
class EmptyInputPlanTests(EquivalenceCase):
    ORDERS_COLUMNS = ("a", "cid", "g", "id", "note", "seq")

    def _catalogs_and_tables(self):
        customers_rows = DATA["small_big"]["customers"]
        customer_columns = _columns_of(customers_rows)
        indexed, plain = Catalog(), Catalog()
        # Index only the NON-empty side: the empty orders table then drives zero lookups as the
        # INLJ outer. An index on the empty side is useless and muddies which candidate won.
        for catalog, customer_indexes in ((indexed, ("cid",)), (plain, ())):
            catalog.add(TableInfo("orders", self.ORDERS_COLUMNS, 0, {}, ()))
            catalog.add(TableInfo(
                "customers", customer_columns, len(customers_rows),
                {column: _distinct(customers_rows, column) for column in customer_columns},
                customer_indexes,
            ))
        tables = {
            "orders": Table("orders", self.ORDERS_COLUMNS, []),
            "customers": Table("customers", customer_columns,
                               [dict(row) for row in customers_rows]),
        }
        return indexed, plain, tables

    def test_empty_outer_agrees_between_inlj_and_hash_with_full_header(self) -> None:
        sql = "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.cid"
        indexed, plain, tables = self._catalogs_and_tables()
        plan_a = plan(parse(sql), indexed)
        plan_b = plan(parse(sql), plain)
        result_a = execute(plan_a.physical, tables)
        result_b = execute(plan_b.physical, tables)
        join_a, join_b = _join_node(plan_a), _join_node(plan_b)
        if not (isinstance(join_a, IndexNestedLoopJoin) and join_a.outer_side == "left"):
            self.fail(_render_report(sql, DATA["small_big"], ["customers.cid"],
                                     _plan_summary(join_a), _plan_summary(join_b), result_a, result_b,
                                     extra="index on the non-empty side failed to drive an empty outer INLJ"))
        if not isinstance(join_b, HashJoin):
            self.fail(_render_report(sql, DATA["small_big"], [],
                                     _plan_summary(join_a), _plan_summary(join_b), result_a, result_b,
                                     extra="baseline failed to stay a hash-join"))
        header = ["orders.a", "orders.cid", "orders.g", "orders.id", "orders.note", "orders.seq",
                  "customers.cid", "customers.name", "customers.seq", "customers.tier", "customers.v"]
        self.assertEqual(result_a[0], header)
        self.assertEqual(result_b[0], header)
        self.assertEqual(result_a, (header, []))
        self.assertEqual(result_b, (header, []))

        # a global aggregate over the same empty join still emits exactly one zero/null group
        aggregate_sql = "SELECT count(*) AS n FROM orders INNER JOIN customers ON orders.cid = customers.cid"
        aggregated_a = execute(plan(parse(aggregate_sql), indexed).physical, tables)
        aggregated_b = execute(plan(parse(aggregate_sql), plain).physical, tables)
        self.assertEqual(aggregated_a, (["n"], [{"n": 0}]))
        self.assertEqual(aggregated_a, aggregated_b)


# -------------------------------------------------------------------------------------------------
# Cross-check the harness itself: an unparseable/unsupported statement must surface as an error
# document, never as a silent "equivalent" -- unsupported syntax is never mixed into the cases.
# -------------------------------------------------------------------------------------------------
class HarnessIntegrityTests(EquivalenceCase):
    def test_unsupported_syntax_is_rejected_not_compared(self) -> None:
        for sql in (
            "SELECT nope FROM events",                                  # unknown column
            "SELECT id FROM orders LEFT JOIN customers ON 1=1",         # unsupported join form
        ):
            with self.subTest(sql=sql):
                code, _, err = self._cli(self._argv("reconcile", "pairs", sql, with_index=True))
                self.assertEqual(code, 2)
                document = json.loads(err)
                self.assertIn(document["error"], ("plan_error", "parse_error", "validation_error"))


if __name__ == "__main__":
    unittest.main()
