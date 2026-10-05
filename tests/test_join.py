"""Two-table equality INNER JOIN: resolution, pushdown, cost choice, and identical execution."""

from __future__ import annotations

import unittest

from sqlplan.parser import parse
from sqlplan.errors import PlanError
from sqlplan.executor import Table, execute
from sqlplan.planner import (
    Catalog,
    HashJoin,
    IndexNestedLoopJoin,
    TableInfo,
    describe_node,
    plan,
)

ORDERS_ROWS = [
    {"id": 1, "cid": 10, "amount": 100.0, "note": "x"},
    {"id": 2, "cid": 20, "amount": 200.0, "note": "y"},
    {"id": 3, "cid": 10, "amount": 300.0, "note": "z"},
    {"id": 4, "cid": None, "amount": 50.0, "note": "q"},
    {"id": 5, "cid": 30, "amount": 400.0, "note": "w"},
]
CUSTOMERS_ROWS = [
    {"cid": 10, "name": "a"},
    {"cid": 20, "name": "b"},
    {"id_only": 0},  # exercises rows that miss columns; pruned scan must tolerate that
    {"cid": 10, "name": "a2"},
    {"cid": 40, "name": "d"},
]

JOIN_SQL = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON orders.cid = customers.cid"


def catalog(left_indexes: tuple[str, ...] = (), right_indexes: tuple[str, ...] = ()) -> Catalog:
    built = Catalog()
    built.add(TableInfo("orders", ("amount", "cid", "id", "note"), len(ORDERS_ROWS), {"id": 5, "cid": 3}, left_indexes))
    built.add(TableInfo("customers", ("cid", "id_only", "name"), len(CUSTOMERS_ROWS), {"cid": 3}, right_indexes))
    return built


def tables() -> dict[str, Table]:
    return {
        "orders": Table("orders", ("amount", "cid", "id", "note"), [dict(row) for row in ORDERS_ROWS]),
        "customers": Table("customers", ("cid", "id_only", "name"), [dict(row) for row in CUSTOMERS_ROWS]),
    }


def join_node(result) -> object:
    return result.physical.input


class JoinValidationTests(unittest.TestCase):
    def test_ambiguous_unqualified_column_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT cid FROM orders INNER JOIN customers ON orders.cid = customers.cid"), catalog())

    def test_unknown_column_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT orders.nope FROM orders INNER JOIN customers ON orders.cid = customers.cid"), catalog())

    def test_unknown_table_qualifier_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT z.id FROM orders INNER JOIN customers ON orders.cid = customers.cid"), catalog())

    def test_qualified_column_not_in_that_table_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT customers.id FROM orders INNER JOIN customers ON orders.cid = customers.cid"), catalog())

    def test_non_equality_on_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT orders.id FROM orders INNER JOIN customers ON orders.cid > customers.cid"), catalog())

    def test_join_columns_from_one_input_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = orders.id"), catalog())

    def test_two_joins_are_a_plan_error(self) -> None:
        built = catalog()
        built.add(TableInfo("extra", ("cid",), 1))
        with self.assertRaises(PlanError):
            plan(
                parse(
                    "SELECT orders.id FROM orders "
                    "INNER JOIN customers ON orders.cid = customers.cid "
                    "INNER JOIN extra ON extra.cid = customers.cid"
                ),
                built,
            )

    def test_self_join_is_a_plan_error(self) -> None:
        with self.assertRaises(PlanError):
            plan(parse("SELECT orders.id FROM orders INNER JOIN orders ON orders.id = orders.cid"), catalog())


class JoinPlanningTests(unittest.TestCase):
    def test_default_plan_is_a_hash_join_with_keys_and_order(self) -> None:
        result = plan(parse(JOIN_SQL), catalog())
        node = join_node(result)
        self.assertIsInstance(node, HashJoin)
        self.assertEqual((node.left_table, node.right_table, node.left_key, node.right_key),
                         ("orders", "customers", "cid", "cid"))
        document = describe_node(node)
        self.assertEqual(document["operator"], "hash-join")
        self.assertEqual(document["leftTable"], "orders")
        self.assertEqual(document["rightTable"], "customers")
        self.assertEqual(document["leftKey"], "cid")
        self.assertEqual(document["rightKey"], "cid")
        self.assertIn("joinOrder", document)
        self.assertIn("rows", document)
        self.assertIn("cost", document)

    def test_hash_builds_the_smaller_filtered_side(self) -> None:
        # customers has 5 rows, orders 5; a pushed equality pins orders to ~1 row, so orders builds
        sql = (
            "SELECT orders.id, customers.name FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid WHERE orders.id = 2"
        )
        result = plan(parse(sql), catalog())
        self.assertIsInstance(join_node(result), HashJoin)
        self.assertEqual(join_node(result).build_side, "left")

    def test_index_nested_loop_wins_when_it_is_cheaper(self) -> None:
        # Small customers outer, very large orders with an index on its join key: probing beats hashing.
        built = Catalog()
        built.add(TableInfo("customers", ("cid", "name"), 3, {"cid": 3}))
        built.add(TableInfo("orders", ("id", "cid"), 10000, {"id": 10000, "cid": 5000}, ("cid",)))
        sql = "SELECT customers.name, orders.id FROM customers INNER JOIN orders ON customers.cid = orders.cid"
        result = plan(parse(sql), built)
        node = join_node(result)
        self.assertIsInstance(node, IndexNestedLoopJoin)
        self.assertEqual(node.outer_side, "left")
        document = describe_node(node)
        self.assertEqual(document["operator"], "index-nested-loop-join")
        self.assertEqual(document["inner"]["operator"], "index-lookup")
        self.assertEqual(document["inner"]["column"], "cid")

    def test_tie_prefers_hash_join(self) -> None:
        # Even with an index available, identical cost keeps hash (index candidate must be strictly cheaper).
        node = join_node(plan(parse(JOIN_SQL), catalog(right_indexes=("cid",))))
        self.assertIsInstance(node, HashJoin)

    def test_one_sided_where_conjuncts_are_pushed_down(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE customers.name = 'a' AND orders.amount > 100"
        )
        result = plan(parse(sql), catalog())
        self.assertTrue(any("moved into customers" in note for note in result.notes))
        self.assertTrue(any("moved into orders" in note for note in result.notes))

    def test_cross_side_conjunct_stays_above_the_join(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.id = customers.id_only"
        )
        result = plan(parse(sql), catalog())
        self.assertTrue(any("stay above the join" in note for note in result.notes))

    def test_columns_are_pruned_per_side(self) -> None:
        result = plan(parse(JOIN_SQL), catalog())
        hash_join = join_node(result)
        left_scan = _leaf(hash_join.left)
        right_scan = _leaf(hash_join.right)
        self.assertEqual(set(left_scan.columns), {"id", "cid"})
        self.assertEqual(set(right_scan.columns), {"cid", "name"})
        self.assertTrue(any("orders scan reads" in note for note in result.notes))
        self.assertTrue(any("customers scan reads" in note for note in result.notes))


class JoinExecutionTests(unittest.TestCase):
    def test_hash_and_index_nested_loop_agree(self) -> None:
        statement = parse(JOIN_SQL)
        hashed = execute(plan(statement, catalog()).physical, tables())
        indexed = execute(plan(statement, catalog(right_indexes=("cid",))).physical, tables())
        self.assertEqual(hashed, indexed)

    def test_duplicate_keys_produce_the_full_combination(self) -> None:
        _, rows = execute(plan(parse(JOIN_SQL), catalog()).physical, tables())
        # orders cid=10 (rows 1,3) x customers cid=10 (a, a2) = four pairs, plus cid=20 = one
        self.assertEqual(len(rows), 5)
        self.assertEqual([(row["id"], row["name"]) for row in rows],
                         [(1, "a"), (1, "a2"), (2, "b"), (3, "a"), (3, "a2")])

    def test_null_join_key_matches_nothing(self) -> None:
        _, rows = execute(plan(parse(JOIN_SQL), catalog()).physical, tables())
        self.assertFalse(any(row["id"] == 4 for row in rows))

    def test_order_is_left_original_then_right_original_regardless_of_build_side(self) -> None:
        build_right = execute(plan(parse(JOIN_SQL), catalog()).physical, tables())
        # force the opposite build direction by making the left side tiny after a pushed filter
        flipped_sql = (
            "SELECT orders.id, customers.name FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid WHERE orders.id IN (1, 3)"
        )
        flipped = execute(plan(parse(flipped_sql), catalog()).physical, tables())
        right_ids = [row["id"] for row in flipped[1]]
        self.assertEqual(right_ids, [1, 1, 3, 3])

    def test_star_expands_to_qualified_columns(self) -> None:
        sql = "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.cid WHERE orders.id = 2"
        columns, rows = execute(plan(parse(sql), catalog(right_indexes=("cid",))).physical, tables())
        # star expands to every pruned column, each qualified, in each side's (sorted) scan order
        self.assertEqual(
            columns,
            [
                "orders.amount",
                "orders.cid",
                "orders.id",
                "orders.note",
                "customers.cid",
                "customers.id_only",
                "customers.name",
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(list(row.keys()) == columns for row in rows))

    def test_star_columns_survive_a_join_that_matches_nothing(self) -> None:
        sql = "SELECT * FROM orders INNER JOIN customers ON orders.cid = customers.cid WHERE orders.id < 0"
        columns, rows = execute(plan(parse(sql), catalog()).physical, tables())
        self.assertEqual(
            columns,
            [
                "orders.amount",
                "orders.cid",
                "orders.id",
                "orders.note",
                "customers.cid",
                "customers.id_only",
                "customers.name",
            ],
        )
        self.assertEqual(rows, [])
        self.assertNotIn("*", columns)

    def test_star_columns_agree_between_hash_and_index_nested_loop_on_zero_rows(self) -> None:
        indexed = Catalog()
        indexed.add(TableInfo("customers", ("cid", "name"), 2, {"cid": 2}))
        indexed.add(TableInfo("orders", ("id", "cid", "amount"), 40, {"cid": 2, "id": 40}, ("cid",)))
        plain = Catalog()
        plain.add(TableInfo("customers", ("cid", "name"), 2, {"cid": 2}))
        plain.add(TableInfo("orders", ("id", "cid", "amount"), 40, {"cid": 2, "id": 40}))
        joined = {
            "customers": Table("customers", ("cid", "name"), [{"cid": 1, "name": "a"}, {"cid": 2, "name": "b"}]),
            "orders": Table(
                "orders",
                ("id", "cid", "amount"),
                [{"id": i, "cid": (i % 2) + 1, "amount": float(i)} for i in range(1, 41)],
            ),
        }
        sql = (
            "SELECT * FROM customers INNER JOIN orders ON customers.cid = orders.cid "
            "WHERE customers.name = 'gone'"
        )
        indexed_plan = plan(parse(sql), indexed)
        plain_plan = plan(parse(sql), plain)
        self.assertIsInstance(indexed_plan.physical.input, IndexNestedLoopJoin)
        self.assertIsInstance(plain_plan.physical.input, HashJoin)
        # neither algorithm produces a row, but both must still publish the full left-then-right schema
        self.assertEqual(execute(indexed_plan.physical, joined), execute(plain_plan.physical, joined))
        columns, rows = execute(indexed_plan.physical, joined)
        self.assertEqual(rows, [])
        self.assertEqual(columns, ["customers.cid", "customers.name", "orders.id", "orders.cid", "orders.amount"])

    def test_star_and_explicit_projections_expand_in_select_position(self) -> None:
        sql = (
            "SELECT orders.id, *, customers.name AS who FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid WHERE orders.id = 2"
        )
        columns, rows = execute(plan(parse(sql), catalog()).physical, tables())
        # the leading explicit column keeps its position, the star fills left-then-right qualified
        # columns in the middle (its duplicate orders.id collapses to the first key), the alias is last
        self.assertEqual(
            columns,
            [
                "id",
                "orders.amount",
                "orders.cid",
                "orders.id",
                "orders.note",
                "customers.cid",
                "customers.id_only",
                "customers.name",
                "who",
            ],
        )
        self.assertTrue(all(list(row.keys()) == columns for row in rows))

    def test_group_and_aggregate_over_a_join(self) -> None:
        sql = (
            "SELECT customers.name, count(*), sum(orders.amount) FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid GROUP BY name ORDER BY name"
        )
        columns, rows = execute(plan(parse(sql), catalog(right_indexes=("cid",))).physical, tables())
        self.assertEqual(columns, ["name", "count(*)", "sum(amount)"])
        self.assertEqual([row["name"] for row in rows], ["a", "a2", "b"])
        self.assertEqual([row["count(*)"] for row in rows], [2, 2, 1])
        self.assertEqual(rows[0]["sum(amount)"], 400.0)

    def test_residual_cross_side_filter_is_applied(self) -> None:
        sql = (
            "SELECT orders.id FROM orders INNER JOIN customers ON orders.cid = customers.cid "
            "WHERE orders.id = customers.id_only"
        )
        _, rows = execute(plan(parse(sql), catalog()).physical, tables())
        self.assertEqual([row["id"] for row in rows], [])

    def test_limit_over_a_join(self) -> None:
        _, rows = execute(
            plan(parse(JOIN_SQL + " ORDER BY orders.id LIMIT 2"), catalog(right_indexes=("cid",))).physical,
            tables(),
        )
        self.assertEqual([(row["id"], row["name"]) for row in rows], [(1, "a"), (1, "a2")])

    def test_pushed_inner_predicate_matches_between_algorithms(self) -> None:
        with_indexes = Catalog()
        with_indexes.add(TableInfo("customers", ("cid", "name"), 2, {"cid": 2}))
        with_indexes.add(TableInfo("orders", ("id", "cid", "amount"), 40, {"cid": 2, "id": 40}, ("cid",)))
        without_indexes = Catalog()
        without_indexes.add(TableInfo("customers", ("cid", "name"), 2, {"cid": 2}))
        without_indexes.add(TableInfo("orders", ("id", "cid", "amount"), 40, {"cid": 2, "id": 40}))
        joined_tables = {
            "customers": Table("customers", ("cid", "name"), [{"cid": 1, "name": "a"}, {"cid": 2, "name": "b"}]),
            "orders": Table("orders", ("id", "cid", "amount"),
                           [{"id": i, "cid": (i % 2) + 1, "amount": float(i)} for i in range(1, 41)]),
        }
        sql = (
            "SELECT customers.name, orders.id FROM customers "
            "INNER JOIN orders ON customers.cid = orders.cid WHERE orders.amount > 35"
        )
        statement = parse(sql)
        plain = execute(plan(statement, without_indexes).physical, joined_tables)
        indexed = execute(plan(statement, with_indexes).physical, joined_tables)
        self.assertIsInstance(plan(statement, with_indexes).physical.input, IndexNestedLoopJoin)
        self.assertEqual(plain, indexed)
        self.assertTrue(all(row["id"] > 35 for row in indexed[1]))


def _leaf(node: object) -> object:
    child = getattr(node, "input", None)
    while child is not None:
        node, child = child, getattr(child, "input", None)
    return node


if __name__ == "__main__":
    unittest.main()
