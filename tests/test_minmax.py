"""min/max semantics: NULL handling, numeric vs string comparison, type preservation and validation.

The rules under test (shared by global, GROUP BY and join aggregates):

* NULLs are ignored; a group with no non-NULL input yields NULL;
* all-number inputs compare numerically (ints and decimals may mix) and the winning row keeps its
  original representation, ties keeping input order;
* all-string inputs compare in the same order ORDER BY uses and return the original string;
* DISTINCT never changes an extremum;
* mixed strings/numbers, Booleans, arrays and objects are rejected with ValidationError, without
  disturbing count/sum/avg in the same query.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import contextlib

from sqlplan.errors import ValidationError
from sqlplan.executor import Table, execute
from sqlplan.parser import parse
from sqlplan.planner import Catalog, TableInfo, plan
from sqlplan.cli import EXIT_ERROR, EXIT_OK, main


def _distinct(rows: list[dict[str, object]], column: str) -> int:
    try:
        return len({row.get(column) for row in rows})
    except TypeError:  # arrays/objects are unhashable; fall back to pairwise equality
        seen: list[object] = []
        for row in rows:
            value = row.get(column)
            if value not in seen:
                seen.append(value)
        return len(seen)


def _build(rows: list[dict[str, object]], table: str = "t") -> tuple[Catalog, Table]:
    columns = tuple(sorted({column for row in rows for column in row}))
    distinct = {column: _distinct(rows, column) for column in columns}
    catalog = Catalog()
    catalog.add(TableInfo(table, columns, len(rows), distinct))
    return catalog, Table(table, columns, [dict(row) for row in rows])


def run_query(sql: str, rows: list[dict[str, object]]):
    catalog, table = _build(rows)
    return execute(plan(parse(sql), catalog).physical, table)


class NumericExtremesTests(unittest.TestCase):
    def test_integer_extrema_stay_integers(self) -> None:
        _, rows = run_query("SELECT min(v), max(v) FROM t", [{"v": 3}, {"v": 1}, {"v": 2}])
        self.assertEqual(rows[0]["min(v)"], 1)
        self.assertEqual(rows[0]["max(v)"], 3)
        self.assertIsInstance(rows[0]["min(v)"], int)
        self.assertIsInstance(rows[0]["max(v)"], int)

    def test_ints_and_decimals_mix_numerically(self) -> None:
        _, rows = run_query("SELECT min(v), max(v) FROM t", [{"v": 2}, {"v": 1.5}, {"v": 4}])
        self.assertEqual(rows[0]["min(v)"], 1.5)
        self.assertEqual(rows[0]["max(v)"], 4)
        self.assertIsInstance(rows[0]["max(v)"], int)

    def test_numeric_tie_keeps_the_first_value_in_input_order(self) -> None:
        # 3 and 3.0 are numerically equal; whichever row is first is the one that is returned.
        _, int_first = run_query("SELECT max(v) FROM t", [{"v": 3}, {"v": 3.0}])
        self.assertIs(int_first[0]["max(v)"], 3)
        _, float_first = run_query("SELECT max(v) FROM t", [{"v": 3.0}, {"v": 3}])
        self.assertEqual(float_first[0]["max(v)"], 3.0)
        self.assertIsInstance(float_first[0]["max(v)"], float)

    def test_nulls_are_ignored_but_no_non_null_means_null(self) -> None:
        rows = [{"g": "a", "v": 5}, {"g": "a", "v": None}, {"g": "b", "v": None}]
        _, out = run_query("SELECT g, min(v), max(v) FROM t GROUP BY g ORDER BY g", rows)
        self.assertEqual(out[0]["g"], "a")
        self.assertEqual(out[0]["min(v)"], 5)
        self.assertEqual(out[0]["max(v)"], 5)
        self.assertIsNone(out[1]["min(v)"])
        self.assertIsNone(out[1]["max(v)"])

    def test_global_aggregate_without_rows_is_null(self) -> None:
        _, rows = run_query("SELECT min(v), max(v), count(*) FROM t WHERE v > 100", [{"v": 1}])
        self.assertIsNone(rows[0]["min(v)"])
        self.assertIsNone(rows[0]["max(v)"])
        self.assertEqual(rows[0]["count(*)"], 0)

    def test_distinct_does_not_change_an_extremum(self) -> None:
        rows = [{"v": 2}, {"v": 2}, {"v": 5}, {"v": 1}]
        _, plain = run_query("SELECT min(v), max(v) FROM t", rows)
        _, distinct = run_query("SELECT min(DISTINCT v), max(DISTINCT v) FROM t", rows)
        self.assertEqual(distinct[0]["min(DISTINCT v)"], plain[0]["min(v)"])
        self.assertEqual(distinct[0]["max(DISTINCT v)"], plain[0]["max(v)"])
        self.assertEqual(distinct[0]["min(DISTINCT v)"], 1)
        self.assertEqual(distinct[0]["max(DISTINCT v)"], 5)


class StringExtremesTests(unittest.TestCase):
    def test_strings_compare_in_sort_order(self) -> None:
        rows = [{"v": "banana"}, {"v": "apple"}, {"v": None}, {"v": "cherry"}]
        _, out = run_query("SELECT min(v), max(v) FROM t", rows)
        self.assertEqual(out[0]["min(v)"], "apple")
        self.assertEqual(out[0]["max(v)"], "cherry")

    def test_duplicate_strings_and_distinct_agree(self) -> None:
        rows = [{"v": "b"}, {"v": "b"}, {"v": "a"}]
        plain = run_query("SELECT min(v), max(v) FROM t", rows)[1]
        distinct = run_query("SELECT min(DISTINCT v), max(DISTINCT v) FROM t", rows)[1]
        self.assertEqual(plain[0]["min(v)"], "a")
        self.assertEqual(distinct[0]["min(DISTINCT v)"], "a")
        self.assertEqual(distinct[0]["max(DISTINCT v)"], "b")

    def test_string_extremes_per_group(self) -> None:
        rows = [{"g": "x", "v": "z"}, {"g": "x", "v": "a"}, {"g": "y", "v": "m"}]
        _, out = run_query("SELECT g, min(v), max(v) FROM t GROUP BY g ORDER BY g", rows)
        self.assertEqual([(r["g"], r["min(v)"], r["max(v)"]) for r in out],
                         [("x", "a", "z"), ("y", "m", "m")])

    def test_aliased_extremes_keep_values(self) -> None:
        columns, out = run_query("SELECT min(v) AS lo, max(v) AS hi FROM t", [{"v": "q"}])
        self.assertEqual(columns, ["lo", "hi"])
        self.assertEqual((out[0]["lo"], out[0]["hi"]), ("q", "q"))


class OtherAggregatesUnaffectedTests(unittest.TestCase):
    def test_nulls_do_not_change_count_sum_avg(self) -> None:
        rows = [{"v": 10}, {"v": None}, {"v": 20}]
        _, out = run_query("SELECT min(v), max(v), count(v), sum(v), avg(v) FROM t", rows)
        record = out[0]
        self.assertEqual(record["min(v)"], 10)
        self.assertEqual(record["max(v)"], 20)
        self.assertEqual(record["count(v)"], 2)
        self.assertEqual(record["sum(v)"], 30.0)
        self.assertEqual(record["avg(v)"], 15.0)

    def test_count_star_still_counts_null_rows_alongside_min(self) -> None:
        _, out = run_query("SELECT min(v), count(*) FROM t", [{"v": None}, {"v": 1}])
        self.assertEqual(out[0]["min(v)"], 1)
        self.assertEqual(out[0]["count(*)"], 2)


class InvalidExtremeInputsTests(unittest.TestCase):
    def test_mixed_numbers_and_strings_raise(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            run_query("SELECT min(v) FROM t", [{"v": 1}, {"v": "x"}])
        self.assertIn("min", caught.exception.message)
        self.assertIn("v", caught.exception.message)

    def test_boolean_is_not_a_number(self) -> None:
        with self.assertRaises(ValidationError):
            run_query("SELECT max(v) FROM t", [{"v": True}, {"v": 3}])
        with self.assertRaises(ValidationError):
            run_query("SELECT min(v) FROM t", [{"v": True}, {"v": False}])

    def test_array_and_object_inputs_raise(self) -> None:
        with self.assertRaises(ValidationError):
            run_query("SELECT min(v) FROM t", [{"v": [1, 2]}, {"v": 3}])
        with self.assertRaises(ValidationError):
            run_query("SELECT max(v) FROM t", [{"v": {"a": 1}}])

    def test_error_names_max_and_column_for_grouped_query(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            run_query("SELECT g, max(v) FROM t GROUP BY g", [{"g": 1, "v": 1}, {"g": 1, "v": "s"}])
        self.assertIn("max", caught.exception.message)
        self.assertEqual(caught.exception.context.get("column"), "v")

    def test_extremum_validates_even_when_sum_is_listed_first(self) -> None:
        # sum(v) must not hit float("x") first; the extremum rule owns the error.
        with self.assertRaises(ValidationError) as caught:
            run_query("SELECT sum(v), count(*), min(v) FROM t", [{"v": 1}, {"v": "x"}])
        self.assertIn("min", caught.exception.message)

    def test_sum_over_strings_is_a_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            run_query("SELECT avg(v) FROM t", [{"v": "a"}])


class JoinExtremesTests(unittest.TestCase):
    def test_string_extremes_over_a_join(self) -> None:
        orders = [
            {"id": 1, "cid": 10, "tag": "zebra"},
            {"id": 2, "cid": 10, "tag": "apple"},
            {"id": 3, "cid": 20, "tag": "mango"},
            {"id": 4, "cid": None, "tag": "pear"},
        ]
        customers = [{"cid": 10, "name": "A"}, {"cid": 20, "name": "B"}]
        catalog = Catalog()
        catalog.add(TableInfo("orders", ("cid", "id", "tag"), 4, {"cid": 2}))
        catalog.add(TableInfo("customers", ("cid", "name"), 2, {"cid": 2}))
        tables = {
            "orders": Table("orders", ("cid", "id", "tag"), [dict(r) for r in orders]),
            "customers": Table("customers", ("cid", "name"), [dict(r) for r in customers]),
        }
        sql = (
            "SELECT customers.name, min(orders.tag), max(orders.tag) FROM orders "
            "INNER JOIN customers ON orders.cid = customers.cid GROUP BY name ORDER BY name"
        )
        _, rows = execute(plan(parse(sql), catalog).physical, tables)
        self.assertEqual(
            [(r["name"], r["min(tag)"], r["max(tag)"]) for r in rows],
            [("A", "apple", "zebra"), ("B", "mango", "mango")],
        )

    def test_indexed_and_plain_plans_agree_on_numeric_extremes(self) -> None:
        rows = [{"id": i, "g": "x" if i % 2 else "y", "v": i} for i in range(1, 21)]
        sql = "SELECT g, min(v), max(v) FROM t GROUP BY g ORDER BY g"
        indexed_catalog = Catalog()
        indexed_catalog.add(TableInfo("t", ("g", "id", "v"), 20, {"g": 2, "id": 20}, ("g",)))
        plain_catalog = Catalog()
        plain_catalog.add(TableInfo("t", ("g", "id", "v"), 20, {"g": 2, "id": 20}))
        table = Table("t", ("g", "id", "v"), [dict(r) for r in rows])
        first = execute(plan(parse(sql), indexed_catalog).physical, table)
        second = execute(plan(parse(sql), plain_catalog).physical, table)
        self.assertEqual(first, second)
        self.assertEqual([(r["g"], r["min(v)"], r["max(v)"]) for r in first[1]],
                         [("x", 1, 19), ("y", 2, 20)])
        self.assertIsInstance(first[1][0]["min(v)"], int)


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class MinMaxCliValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "data.jsonl")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, rows: list[dict[str, object]]) -> str:
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return self.path

    def test_run_on_mixed_types_is_a_single_validation_error(self) -> None:
        path = self.write([{"v": 1}, {"v": "x"}])
        code, out, err = _run_cli(["run", "--sql", "SELECT max(v) FROM data", "--table", path])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertIn("max", document["message"])
        self.assertIn("v", document["message"])

    def test_reconcile_on_mixed_types_is_validation_error_exit_two(self) -> None:
        path = self.write([{"v": 1}, {"v": "x"}])
        code, out, err = _run_cli(["reconcile", "--sql", "SELECT min(v) FROM data", "--table", path])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_array_column_does_not_leak_typeerror_and_leaves_output_untouched(self) -> None:
        path = self.write([{"v": [1, 2]}, {"v": 3}])
        target = os.path.join(self.directory.name, "out.jsonl")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("KEEP-ME")
        code, out, err = _run_cli(
            ["run", "--sql", "SELECT min(v) FROM data", "--table", path, "--output", target]
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")
        with open(target, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "KEEP-ME")

    def test_valid_string_extreme_run_succeeds(self) -> None:
        path = self.write([{"v": "b"}, {"v": "a"}, {"v": None}])
        code, out, err = _run_cli(["run", "--sql", "SELECT min(v), max(v) FROM data", "--table", path])
        self.assertEqual((code, err), (EXIT_OK, ""))
        rows = json.loads(out)["rows"]
        self.assertEqual((rows[0]["min(v)"], rows[0]["max(v)"]), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
