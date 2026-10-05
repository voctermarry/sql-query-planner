"""Two-table equi INNER JOIN: parsing, cost-based join choice, deterministic execution, CLI."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from sqlplan import parse
from sqlplan.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from sqlplan.errors import ParseError, PlanError
from sqlplan.executor import Table, execute
from sqlplan.parser import Column, Comparison, Join, Literal
from sqlplan.planner import (
    Catalog,
    HashJoin,
    IndexNestedLoop,
    PhysicalFilter,
    TableInfo,
    join_summary,
    plan,
)

ORDERS = [
    {"id": 1, "cust": 10, "amount": 5.0},
    {"id": 2, "cust": 20, "amount": 7.5},
    {"id": 3, "cust": 10, "amount": 1.5},
    {"id": 4, "cust": None, "amount": 9.0},
]
CUSTOMERS = [
    {"cid": 10, "name": "ada"},
    {"cid": 10, "name": "ada-dup"},
    {"cid": 30, "name": "grace"},
]
# a right side big enough that probing it row by row loses to an index lookup
BIG = [{"bid": index, "payload": f"p{index}"} for index in range(1, 201)]


def catalog(*, right_rows=CUSTOMERS, right_name="customers", right_key="cid", right_indexes=(), left_indexes=()) -> Catalog:
    built = Catalog()
    built.add(
        TableInfo(
            name="orders",
            columns=("amount", "cust", "id"),
            rows=len(ORDERS),
            distinct={"id": 4, "cust": 3, "amount": 4},
            indexes=left_indexes,
        )
    )
    columns = tuple(sorted(right_rows[0]))
    built.add(
        TableInfo(
            name=right_name,
            columns=columns,
            rows=len(right_rows),
            distinct={column: len({row.get(column) for row in right_rows}) for column in columns},
            indexes=right_indexes,
        )
    )
    return built


def tables(*, right_rows=CUSTOMERS, right_name="customers") -> dict[str, Table]:
    return {
        "orders": Table("orders", ("amount", "cust", "id"), [dict(row) for row in ORDERS]),
        right_name: Table(right_name, tuple(sorted(right_rows[0])), [dict(row) for row in right_rows]),
    }


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class JoinParserTests(unittest.TestCase):
    def test_join_parses_into_the_statement(self) -> None:
        statement = parse("SELECT orders.id FROM orders INNER JOIN customers ON orders.cust = customers.cid")
        self.assertEqual(statement.table, "orders")
        self.assertEqual(len(statement.joins), 1)
        join = statement.joins[0]
        self.assertEqual(join.table, "customers")
        self.assertEqual(join.condition, Comparison(Column("cust", "orders"), "=", Column("cid", "customers")))

    def test_join_appears_in_the_document(self) -> None:
        document = parse("SELECT id FROM orders INNER JOIN customers ON cust = cid").to_document()
        self.assertEqual(document["joins"][0]["table"], "customers")
        self.assertEqual(document["joins"][0]["on"]["kind"], "comparison")

    def test_broken_on_condition_is_a_parse_error_with_a_position(self) -> None:
        with self.assertRaises(ParseError) as caught:
            parse("SELECT id FROM orders INNER JOIN customers ON orders.cust =")
        self.assertEqual(caught.exception.kind, "parse_error")
        self.assertIn("line", caught.exception.context)
        self.assertIn("column", caught.exception.context)

    def test_missing_on_is_a_parse_error(self) -> None:
        with self.assertRaises(ParseError):
            parse("SELECT id FROM orders INNER JOIN customers")

    def test_two_joins_parse_so_the_planner_can_reject_them(self) -> None:
        statement = parse("SELECT id FROM a INNER JOIN b ON a.x = b.x INNER JOIN c ON b.x = c.x")
        self.assertEqual(len(statement.joins), 2)


class JoinPlanErrorTests(unittest.TestCase):
    def test_multiple_joins_are_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM orders INNER JOIN customers ON cust = cid INNER JOIN customers ON cust = cid"), catalog())

    def test_self_join_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError) as caught:
            plan(parse("SELECT orders.id FROM orders INNER JOIN orders ON orders.id = orders.id"), catalog())
        self.assertIn("self", str(caught.exception))

    def test_non_equality_condition_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM orders INNER JOIN customers ON orders.cust > customers.cid"), catalog())

    def test_join_against_a_literal_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM orders INNER JOIN customers ON orders.cust = 5"), catalog())

    def test_join_keys_from_one_side_are_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM orders INNER JOIN customers ON orders.cust = orders.id"), catalog())

    def test_ambiguous_unqualified_column_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError) as caught:
            plan(parse("SELECT cust FROM orders INNER JOIN customers ON orders.cust = customers.cid"), catalog(right_rows=[{"cid": 1, "cust": 1, "name": "x"}]))
        self.assertIn("ambiguous", str(caught.exception))

    def test_unknown_table_qualifier_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT nope.id FROM orders INNER JOIN customers ON orders.cust = customers.cid"), catalog())

    def test_unknown_column_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT orders.zzz FROM orders INNER JOIN customers ON orders.cust = customers.cid"), catalog())

    def test_unknown_join_table_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT id FROM orders INNER JOIN missing ON orders.cust = missing.cid"), catalog())


class JoinChoiceTests(unittest.TestCase):
    SQL = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON orders.cust = customers.cid"

    def physical_join(self, sql: str, cat: Catalog):
        node = plan(parse(sql), cat).physical
        while not isinstance(node, (HashJoin, IndexNestedLoop)):
            node = node.input
        return node

    def test_hash_join_builds_the_smaller_side(self) -> None:
        join = self.physical_join(self.SQL, catalog())
        self.assertIsInstance(join, HashJoin)
        # customers (3 rows) is smaller than orders (4): building on it is cheaper
        self.assertEqual(join.build, "right")

    def test_cost_tie_prefers_the_from_order_build_side(self) -> None:
        join = self.physical_join(self.SQL, catalog(right_rows=CUSTOMERS[:4] if len(CUSTOMERS) >= 4 else CUSTOMERS + [{"cid": 40, "name": "d"}]))
        self.assertEqual(join.build, "left")

    def test_hash_join_beats_no_index_and_index_nested_loop_wins_with_one(self) -> None:
        sql = "SELECT orders.id, big.payload FROM orders INNER JOIN big ON orders.cust = big.bid"
        plain = self.physical_join(sql, catalog(right_rows=BIG, right_name="big", right_key="bid"))
        self.assertIsInstance(plain, HashJoin)
        indexed = self.physical_join(sql, catalog(right_rows=BIG, right_name="big", right_key="bid", right_indexes=("bid",)))
        self.assertIsInstance(indexed, IndexNestedLoop)
        self.assertEqual(indexed.inner_table, "big")

    def test_plan_document_exposes_the_join(self) -> None:
        document = plan(parse(self.SQL), catalog()).to_document()
        summary = document["join"]
        self.assertEqual(summary["operator"], "hash-join")
        self.assertEqual(summary["leftKey"], "orders.cust")
        self.assertEqual(summary["rightKey"], "customers.cid")
        self.assertEqual(summary["order"], ["customers", "orders"])
        self.assertIn("estimatedRows", summary)
        self.assertIn("estimatedCost", document)
        physical = document["physical"]
        join_node = physical["input"]
        self.assertEqual(join_node["operator"], "hash-join")
        self.assertIn("left", join_node)
        self.assertIn("right", join_node)

    def test_join_candidates_are_listed_in_the_notes(self) -> None:
        result = plan(parse(self.SQL), catalog())
        self.assertTrue(any("join candidate hash-join build=left" in note for note in result.notes))
        self.assertTrue(any(note.startswith("join: hash-join") for note in result.notes))

    def test_single_side_where_conjuncts_are_pushed_down(self) -> None:
        sql = self.SQL + " WHERE orders.amount > 1 AND customers.name = 'ada'"
        result = plan(parse(sql), catalog())
        join = self.physical_join(sql, catalog())
        self.assertIsInstance(join.left, PhysicalFilter)
        self.assertIsInstance(join.right, PhysicalFilter)
        self.assertTrue(any("predicate pushdown" in note and "orders" in note for note in result.notes))
        self.assertTrue(any("predicate pushdown" in note and "customers" in note for note in result.notes))

    def test_projection_pruning_reads_only_needed_columns_per_side(self) -> None:
        join = self.physical_join(self.SQL, catalog())
        left_scan = join.left.input if isinstance(join.left, PhysicalFilter) else join.left
        right_scan = join.right.input if isinstance(join.right, PhysicalFilter) else join.right
        self.assertEqual(set(left_scan.columns), {"id", "cust"})
        self.assertEqual(set(right_scan.columns), {"cid", "name"})

    def test_cross_table_predicates_stay_above_the_join(self) -> None:
        sql = self.SQL + " WHERE orders.amount > 100 OR customers.name = 'ada'"
        result = plan(parse(sql), catalog())
        self.assertIsInstance(result.physical.input, PhysicalFilter)
        self.assertTrue(any("cross-table" in note for note in result.notes))


class JoinExecutionTests(unittest.TestCase):
    SQL = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON orders.cust = customers.cid"

    def test_duplicate_keys_produce_every_combination_in_canonical_order(self) -> None:
        columns, rows = execute(plan(parse(self.SQL), catalog()).physical, tables())
        self.assertEqual(columns, ["id", "name"])
        # left row order, and within one left row the right table's original order;
        # the null cust on order 4 never matches
        self.assertEqual(
            [(row["id"], row["name"]) for row in rows],
            [(1, "ada"), (1, "ada-dup"), (3, "ada"), (3, "ada-dup")],
        )

    def test_every_physical_algorithm_produces_identical_output(self) -> None:
        sql = "SELECT orders.id, big.payload FROM orders INNER JOIN big ON orders.cust = big.bid"
        small_right = [{"bid": 10, "payload": "p10"}, {"bid": 20, "payload": "p20"}]
        cases = [
            (catalog(right_rows=BIG, right_name="big"), BIG),  # hash, build=left
            (catalog(right_rows=small_right, right_name="big"), small_right),  # hash, build=right
            (catalog(right_rows=BIG, right_name="big", right_indexes=("bid",)), BIG),  # index nested loop
        ]
        results = [execute(plan(parse(sql), cat).physical, tables(right_rows=rows, right_name="big")) for cat, rows in cases]
        self.assertIsInstance(plan(parse(sql), cases[1][0]).physical.input, HashJoin)
        self.assertEqual(plan(parse(sql), cases[1][0]).physical.input.build, "right")
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertEqual([row["id"] for row in results[0][1]], [1, 2, 3])

    def test_join_summary_matches_the_chosen_operator(self) -> None:
        sql = "SELECT orders.id, big.payload FROM orders INNER JOIN big ON orders.cust = big.bid"
        summary = join_summary(plan(parse(sql), catalog(right_rows=BIG, right_name="big", right_indexes=("bid",))).physical)
        self.assertEqual(summary["operator"], "index-nested-loop")
        self.assertEqual(summary["order"], ["orders", "big"])

    def test_group_by_and_order_by_over_a_join(self) -> None:
        sql = (
            "SELECT customers.name, count(*) FROM orders INNER JOIN customers "
            "ON orders.cust = customers.cid GROUP BY customers.name ORDER BY customers.name"
        )
        columns, rows = execute(plan(parse(sql), catalog()).physical, tables())
        self.assertEqual(columns, ["name", "count(*)"])
        self.assertEqual([(row["name"], row["count(*)"]) for row in rows], [("ada", 2), ("ada-dup", 2)])

    def test_star_expands_both_inputs_and_qualifies_shared_names(self) -> None:
        cat = catalog(right_rows=[{"cid": 10, "cust": 10, "name": "ada"}])
        sql = "SELECT * FROM orders INNER JOIN customers ON orders.cust = customers.cid"
        columns, rows = execute(plan(parse(sql), cat).physical, tables(right_rows=[{"cid": 10, "cust": 10, "name": "ada"}]))
        self.assertIn("orders.cust", columns)
        self.assertIn("customers.cust", columns)
        self.assertIn("id", columns)
        self.assertEqual(len(rows), 2)  # orders 1 and 3 match


class JoinCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.paths: dict[str, str] = {}
        for name, rows in (("orders", ORDERS), ("customers", CUSTOMERS), ("big", BIG)):
            path = os.path.join(self.directory.name, f"{name}.jsonl")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            self.paths[name] = path
        self.bindings = [arg for name in ("orders", "customers") for arg in ("--table", f"{name}={self.paths[name]}")]

    def tearDown(self) -> None:
        self.directory.cleanup()

    SQL = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON orders.cust = customers.cid"

    def test_run_a_join_with_named_table_bindings(self) -> None:
        code, out, err = run_cli(["run", "--sql", self.SQL, *self.bindings])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(len(document["rows"]), 4)

    def test_run_with_no_matching_rows_is_a_negative_verdict(self) -> None:
        code, _, _ = run_cli(["run", "--sql", self.SQL + " WHERE customers.name = 'nobody'", *self.bindings])
        self.assertEqual(code, EXIT_NEGATIVE)

    def test_reconcile_reports_join_operator_and_order_for_both_plans(self) -> None:
        code, out, _ = run_cli(
            [
                "reconcile",
                "--sql",
                "SELECT orders.id, big.payload FROM orders INNER JOIN big ON orders.cust = big.bid",
                "--table",
                f"orders={self.paths['orders']}",
                "--table",
                f"big={self.paths['big']}",
                "--index",
                "big.bid",
            ]
        )
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["identical"])
        with_index, without_index = document["plans"]
        self.assertEqual(with_index["joinOperator"], "index-nested-loop")
        self.assertEqual(without_index["joinOperator"], "hash-join")
        self.assertEqual(with_index["joinOrder"], ["orders", "big"])

    def test_missing_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(["run", "--sql", self.SQL, "--table", f"orders={self.paths['orders']}"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_duplicate_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(["run", "--sql", self.SQL, *self.bindings, "--table", f"orders={self.paths['orders']}"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_malformed_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(["run", "--sql", self.SQL, "--table", "orders="])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_bare_table_path_is_rejected_for_a_join(self) -> None:
        code, _, err = run_cli(["run", "--sql", self.SQL, "--table", self.paths["orders"], "--table", self.paths["customers"]])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_ambiguous_unqualified_index_is_a_validation_error(self) -> None:
        # "cust" exists on both sides here, so a bare --index cust cannot be resolved
        shared = os.path.join(self.directory.name, "shared.jsonl")
        with open(shared, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"cid": 10, "cust": 10}) + "\n")
        code, _, err = run_cli(
            [
                "plan",
                "--sql",
                "SELECT orders.id FROM orders INNER JOIN shared ON orders.cust = shared.cid",
                "--table",
                f"orders={self.paths['orders']}",
                "--table",
                f"shared={shared}",
                "--index",
                "cust",
            ]
        )
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")

    def test_output_colliding_with_any_input_is_rejected_before_reading(self) -> None:
        for target in (self.paths["orders"], self.paths["customers"]):
            with open(target, encoding="utf-8") as handle:
                before = handle.read()
            code, _, err = run_cli(["run", "--sql", self.SQL, *self.bindings, "--output", target])
            self.assertEqual(code, EXIT_ERROR)
            self.assertEqual(json.loads(err)["error"], "output_error")
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), before)

    def test_single_table_form_still_works(self) -> None:
        code, out, _ = run_cli(["run", "--sql", "SELECT id FROM orders WHERE cust = 10", "--table", self.paths["orders"], "--index", "cust"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual([row["id"] for row in json.loads(out)["rows"]], [1, 3])

    def test_describe_advertises_the_join_operators(self) -> None:
        code, out, _ = run_cli(["describe"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertIn("hash-join", document["operators"])
        self.assertIn("index-nested-loop", document["operators"])
        self.assertFalse(any("no join planning" in item for item in document["extensionSurface"]))


if __name__ == "__main__":
    unittest.main()
