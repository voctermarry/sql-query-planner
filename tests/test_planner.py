"""Planner and executor: cost-based choices, rewrites that are visible, and agreeing plans."""

from __future__ import annotations

import unittest

from sqlplan import parse
from sqlplan.errors import PlanError, ValidationError
from sqlplan.executor import Table, execute
from sqlplan.parser import Column, Comparison, Literal
from sqlplan.planner import Catalog, FullScan, IndexScan, TableInfo, estimate_selectivity, explain, plan, referenced_columns, rows_produced

ROWS = [
    {"id": 1, "region": "eu", "amount": 10.0},
    {"id": 2, "region": "us", "amount": 20.0},
    {"id": 3, "region": "eu", "amount": 30.0},
    {"id": 4, "region": "apac", "amount": None},
    {"id": 5, "region": "eu", "amount": 40.0},
]


def catalog(indexed: bool = True) -> Catalog:
    # `Catalog.add` mutates and returns None (deliberate: a mutator that also returns self reads like a
    # builder), so build the object first. Getting this wrong produced 18 errors on the first run.
    built = Catalog()
    built.add(
        TableInfo(
            name="orders",
            columns=("id", "region", "amount"),
            rows=len(ROWS),
            distinct={"id": len(ROWS), "region": 3},
            indexes=("region",) if indexed else (),
        )
    )
    return built


def table() -> Table:
    return Table("orders", ("id", "region", "amount"), [dict(row) for row in ROWS])


class CatalogueTests(unittest.TestCase):
    def test_unknown_table_is_reported(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM missing"), Catalog())

    def test_index_on_unknown_column_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TableInfo("t", ("a",), 1, indexes=("b",))


class EstimateTests(unittest.TestCase):
    def test_equality_uses_distinct_counts(self) -> None:
        info = TableInfo("t", ("region",), 100, distinct={"region": 4})
        self.assertAlmostEqual(estimate_selectivity(Comparison(Column("region"), "=", Literal("eu")), info), 0.25)

    def test_equality_without_statistics_uses_the_default(self) -> None:
        info = TableInfo("t", ("region",), 100)
        self.assertAlmostEqual(estimate_selectivity(Comparison(Column("region"), "=", Literal("eu")), info), 0.1)

    def test_and_multiplies_and_or_combines(self) -> None:
        statement = parse("SELECT id FROM orders WHERE region = 'eu' AND id = 3 OR region = 'us'")
        info = TableInfo("orders", ("id", "region"), 100, distinct={"id": 100, "region": 4})
        value = estimate_selectivity(statement.where, info)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)


class PlanTests(unittest.TestCase):
    def test_projection_pruning_is_reported_and_applied(self) -> None:
        result = plan(parse("SELECT region FROM orders"), catalog())
        self.assertIsInstance(result.physical, __import__("sqlplan.planner", fromlist=["PhysicalProject"]).PhysicalProject)
        scan = result.physical.input
        self.assertEqual(scan.columns, ("region",))
        self.assertTrue(any("projection pruning" in note for note in result.notes))

    def test_index_scan_is_chosen_when_the_index_pays_off(self) -> None:
        result = plan(parse("SELECT id FROM orders WHERE region = 'eu'"), catalog(indexed=True))
        self.assertIsInstance(result.physical.input, IndexScan)
        self.assertTrue(any("index scan" in note for note in result.notes))

    def test_full_scan_is_kept_when_there_is_no_index(self) -> None:
        result = plan(parse("SELECT id FROM orders WHERE region = 'eu'"), catalog(indexed=False))
        self.assertIsInstance(result.physical.input.input, FullScan)

    def test_global_aggregate_is_noted(self) -> None:
        result = plan(parse("SELECT count(*) FROM orders"), catalog())
        self.assertTrue(any("global aggregate" in note for note in result.notes))

    def test_group_by_aggregate_is_noted(self) -> None:
        result = plan(parse("SELECT region, sum(amount) FROM orders GROUP BY region"), catalog())
        self.assertTrue(any("aggregate:" in note for note in result.notes))

    def test_limit_is_noted(self) -> None:
        result = plan(parse("SELECT id FROM orders LIMIT 2"), catalog())
        self.assertTrue(any("limit:" in note for note in result.notes))
        self.assertEqual(result.estimated_rows, 2)

    def test_unknown_column_is_a_plan_error_with_its_clause(self) -> None:
        with self.assertRaises(PlanError) as caught:
            plan(parse("SELECT nope FROM orders"), catalog())
        self.assertIn("projection", str(caught.exception.context.get("in_clause")))

    def test_star_reads_every_column(self) -> None:
        result = plan(parse("SELECT * FROM orders"), catalog())
        self.assertEqual(result.physical.input.columns, ("id", "region", "amount"))

    def test_explain_is_json_shaped_and_stable(self) -> None:
        document = explain(plan(parse("SELECT region, count(*) FROM orders GROUP BY region ORDER BY region DESC LIMIT 3"), catalog()))
        self.assertEqual(document["table"], "orders")
        self.assertEqual(document["physical"]["operator"], "limit")
        self.assertEqual(document["physical"]["input"]["operator"], "sort")
        self.assertIn("logical", document)

    def test_referenced_columns_walks_aggregates_and_booleans(self) -> None:
        statement = parse("SELECT sum(amount) FROM orders WHERE region = 'eu' OR id IN (1, 2)")
        names = referenced_columns(statement.projections[0].expression) | referenced_columns(statement.where)
        self.assertEqual(names, {"amount", "region", "id"})


class ExecutionTests(unittest.TestCase):
    def run_query(self, sql: str):
        return execute(plan(parse(sql), catalog()).physical, table())

    def test_full_scan_and_projection(self) -> None:
        columns, rows = self.run_query("SELECT id, region FROM orders")
        self.assertEqual(columns, ["id", "region"])
        self.assertEqual(len(rows), len(ROWS))

    def test_index_scan_returns_the_same_rows_as_a_full_scan(self) -> None:
        indexed = execute(plan(parse("SELECT id FROM orders WHERE region = 'eu'"), catalog(indexed=True)).physical, table())
        plain = execute(plan(parse("SELECT id FROM orders WHERE region = 'eu'"), catalog(indexed=False)).physical, table())
        self.assertEqual(indexed, plain)
        self.assertEqual([row["id"] for row in indexed[1]], [1, 3, 5])

    def test_group_by_aggregates(self) -> None:
        columns, rows = self.run_query("SELECT region, count(*), sum(amount) FROM orders GROUP BY region ORDER BY region")
        self.assertEqual(columns, ["region", "count(*)", "sum(amount)"])
        self.assertEqual([row["region"] for row in rows], ["apac", "eu", "us"])
        self.assertEqual([row["count(*)"] for row in rows], [1, 3, 1])
        self.assertEqual(rows[1]["sum(amount)"], 80.0)

    def test_null_is_ignored_by_sum_but_counted_by_count_star(self) -> None:
        _, rows = self.run_query("SELECT region, sum(amount), count(*) FROM orders WHERE region = 'apac' GROUP BY region")
        self.assertIsNone(rows[0]["sum(amount)"])
        self.assertEqual(rows[0]["count(*)"], 1)

    def test_distinct_count(self) -> None:
        _, rows = self.run_query("SELECT count(DISTINCT region) FROM orders")
        self.assertEqual(rows[0]["count(DISTINCT region)"], 3)

    def test_where_with_in_and_is_null(self) -> None:
        _, rows = self.run_query("SELECT id FROM orders WHERE amount IS NULL")
        self.assertEqual([row["id"] for row in rows], [4])
        _, rows = self.run_query("SELECT id FROM orders WHERE id IN (2, 4)")
        self.assertEqual(sorted(row["id"] for row in rows), [2, 4])

    def test_order_by_descending_and_limit(self) -> None:
        _, rows = self.run_query("SELECT id FROM orders ORDER BY id DESC LIMIT 2")
        self.assertEqual([row["id"] for row in rows], [5, 4])

    def test_alias_names_the_output_column(self) -> None:
        columns, _ = self.run_query("SELECT region AS area FROM orders LIMIT 1")
        self.assertEqual(columns, ["area"])

    def test_global_aggregate_over_zero_rows_still_returns_one_row(self) -> None:
        _, rows = self.run_query("SELECT count(*) FROM orders WHERE id > 100")
        self.assertEqual(rows, [{"count(*)": 0}])


if __name__ == "__main__":
    unittest.main()
