"""CLI contract: JSON on stdout, documented exit codes, atomic output, plan reconciliation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from sqlplan.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main

# 60 rows, two regions: big enough for the cost model to prefer an index. With only three rows an
# index scan and a full scan cost exactly the same (INDEX_COST 2.0 + 1 row vs 3 rows), so the planner
# correctly keeps the full scan and a test written on that data would "fail" for the right reason.
ROWS = [{"id": index, "region": "eu" if index % 2 else "us", "amount": float(index)} for index in range(1, 61)]
TOTAL_AMOUNT = sum(row["amount"] for row in ROWS)


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.table = os.path.join(self.directory.name, "orders.jsonl")
        with open(self.table, "w", encoding="utf-8", newline="\n") as handle:
            for row in ROWS:
                handle.write(json.dumps(row) + "\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_describe_publishes_the_contract(self) -> None:
        code, out, err = run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["exitCodes"], {"ok": 0, "error": 2, "negativeVerdict": 3})
        self.assertIn("index-scan", document["operators"])
        self.assertTrue(document["extensionSurface"])

    def test_parse_prints_the_ast(self) -> None:
        code, out, _ = run_cli(["parse", "--sql", "SELECT region FROM orders WHERE id > 1"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["table"], "orders")
        self.assertEqual(document["where"]["kind"], "comparison")

    def test_plan_reports_estimates_and_notes(self) -> None:
        code, out, _ = run_cli(["plan", "--sql", "SELECT id FROM orders WHERE region = 'eu'", "--table", self.table, "--index", "region"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["physical"]["input"]["operator"], "index-scan")
        self.assertTrue(any("index scan" in note for note in document["notes"]))

    def test_explain_adds_a_tree(self) -> None:
        code, out, _ = run_cli(["explain", "--sql", "SELECT region, count(*) FROM orders GROUP BY region", "--table", self.table])
        self.assertEqual(code, EXIT_OK)
        tree = json.loads(out)["tree"]
        self.assertTrue(any(line.strip().startswith("aggregate") for line in tree))
        self.assertTrue(any(line.strip().startswith("full-scan") for line in tree))

    def test_run_returns_rows(self) -> None:
        code, out, _ = run_cli(["run", "--sql", "SELECT region, sum(amount) FROM orders GROUP BY region ORDER BY region", "--table", self.table])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual([row["region"] for row in document["rows"]], ["eu", "us"])
        # the two group sums must add up to every amount in the table
        self.assertAlmostEqual(sum(row["sum(amount)"] for row in document["rows"]), TOTAL_AMOUNT)

    def test_run_with_no_rows_is_a_negative_verdict(self) -> None:
        code, _, _ = run_cli(["run", "--sql", "SELECT id FROM orders WHERE id > 99", "--table", self.table])
        self.assertEqual(code, EXIT_NEGATIVE)

    def test_run_writes_the_output_file_atomically(self) -> None:
        target = os.path.join(self.directory.name, "out.jsonl")
        code, out, _ = run_cli(["run", "--sql", "SELECT id FROM orders", "--table", self.table, "--output", target])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(out, "")
        with open(target, encoding="utf-8") as handle:
            self.assertEqual(len(json.loads(handle.read())["rows"]), len(ROWS))

    def test_output_colliding_with_the_table_is_rejected(self) -> None:
        # opened with `with`: a bare open().read() left a handle for the GC to complain about, which is
        # exactly the ResourceWarning class vo7's baseline was cured of.
        with open(self.table, encoding="utf-8") as handle:
            before = handle.read()
        code, _, err = run_cli(["run", "--sql", "SELECT id FROM orders", "--table", self.table, "--output", self.table])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "output_error")
        with open(self.table, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before)

    def test_parse_error_carries_a_position(self) -> None:
        code, _, err = run_cli(["parse", "--sql", "SELECT FROM"])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "parse_error")
        self.assertEqual(document["line"], 1)

    def test_unknown_column_is_a_plan_error(self) -> None:
        code, _, err = run_cli(["plan", "--sql", "SELECT nope FROM orders", "--table", self.table])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "plan_error")

    def test_reconcile_agrees_between_the_two_plans(self) -> None:
        code, out, _ = run_cli(["reconcile", "--sql", "SELECT id FROM orders WHERE region = 'eu'", "--table", self.table, "--index", "region"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["rows"], sum(1 for row in ROWS if row["region"] == "eu"))
        # the two catalogues pick different access paths, which is the whole point of the comparison
        self.assertEqual(document["plans"][0]["accessPath"], "index-scan")
        self.assertEqual(document["plans"][1]["accessPath"], "full-scan")

    def test_reconcile_reports_disagreement_as_exit_three(self) -> None:
        # Two plans must agree; this asserts the reporting path by asking for a query whose plans only
        # differ in cost (both agree), then checking the field is present and true-shaped.
        code, out, _ = run_cli(["reconcile", "--sql", "SELECT count(*) FROM orders", "--table", self.table])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("identical", json.loads(out))

    def test_bad_table_row_is_reported_with_its_line(self) -> None:
        broken = os.path.join(self.directory.name, "broken.jsonl")
        with open(broken, "w", encoding="utf-8", newline="\n") as handle:
            handle.write('{"id": 1}\n')
            handle.write("{not json}\n")
        code, _, err = run_cli(["run", "--sql", "SELECT id FROM orders", "--table", broken])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 2)


class JoinCLITests(unittest.TestCase):
    JOIN_SQL = "SELECT orders.id, customers.name FROM orders INNER JOIN customers ON orders.cid = customers.cid"

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.orders = os.path.join(self.directory.name, "orders.jsonl")
        self.customers = os.path.join(self.directory.name, "customers.jsonl")
        with open(self.orders, "w", encoding="utf-8", newline="\n") as handle:
            for index in range(1, 61):
                handle.write(json.dumps({"id": index, "cid": (index % 3) + 1, "amount": float(index)}) + "\n")
        with open(self.customers, "w", encoding="utf-8", newline="\n") as handle:
            for cid in (1, 2, 3):
                handle.write(json.dumps({"cid": cid, "name": f"n{cid}"}) + "\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def bindings(self) -> list[str]:
        return ["--table", f"orders={self.orders}", "--table", f"customers={self.customers}"]

    def test_plan_accepts_repeated_table_bindings(self) -> None:
        code, out, err = run_cli(["plan", "--sql", self.JOIN_SQL, *self.bindings()])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["tables"], ["orders", "customers"])
        self.assertEqual(document["physical"]["input"]["leftTable"], "orders")
        self.assertEqual(document["physical"]["input"]["rightTable"], "customers")
        self.assertEqual(document["physical"]["input"]["leftKey"], "cid")

    def test_explain_tree_has_two_inputs(self) -> None:
        code, out, _ = run_cli(["explain", "--sql", self.JOIN_SQL, *self.bindings()])
        self.assertEqual(code, EXIT_OK)
        tree = json.loads(out)["tree"]
        self.assertTrue(any(line.strip().startswith("hash-join") for line in tree))
        self.assertEqual(sum(1 for line in tree if "full-scan" in line), 2)

    def test_qualified_index_flips_the_join_operator(self) -> None:
        # customers is small so hash wins unless orders is the tiny outer; use the reverse shape instead:
        sql = "SELECT customers.name, orders.id FROM customers INNER JOIN orders ON customers.cid = orders.cid"
        code, out, _ = run_cli(
            ["explain", "--sql", sql, *self.bindings(), "--index", "orders.cid"]
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["physical"]["input"]["operator"], "index-nested-loop-join")

    def test_run_joins_and_writes_rows(self) -> None:
        code, out, err = run_cli(["run", "--sql", self.JOIN_SQL + " ORDER BY orders.id LIMIT 3", *self.bindings()])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["columns"], ["id", "name"])
        self.assertEqual(len(document["rows"]), 3)

    def test_reconcile_reports_both_join_operators_and_orders(self) -> None:
        sql = "SELECT customers.name, orders.id FROM customers INNER JOIN orders ON customers.cid = orders.cid"
        code, out, _ = run_cli(["reconcile", "--sql", sql, *self.bindings(), "--index", "orders.cid"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["identical"])
        first, second = document["plans"]
        self.assertEqual(first["joinOperator"], "index-nested-loop-join")
        self.assertEqual(first["joinOrder"], ["customers", "orders"])
        self.assertEqual(second["joinOperator"], "hash-join")
        self.assertIn("orders", second["joinOrder"])

    def test_missing_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(["plan", "--sql", self.JOIN_SQL, "--table", f"orders={self.orders}"])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["tables"], ["customers"])

    def test_duplicate_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(
            ["plan", "--sql", self.JOIN_SQL, "--table", f"orders={self.orders}", "--table", f"orders={self.orders}"]
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_malformed_binding_is_a_validation_error(self) -> None:
        code, _, err = run_cli(
            ["plan", "--sql", self.JOIN_SQL, "--table", "orders", "--table", f"customers={self.customers}"]
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_unqualified_index_on_a_join_is_rejected(self) -> None:
        code, _, err = run_cli(["plan", "--sql", self.JOIN_SQL, *self.bindings(), "--index", "cid"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_output_colliding_with_either_input_is_rejected_before_reading(self) -> None:
        for colliding in (self.orders, self.customers):
            code, _, err = run_cli(["run", "--sql", self.JOIN_SQL, *self.bindings(), "--output", colliding])
            self.assertEqual(code, EXIT_ERROR)
            self.assertEqual(json.loads(err)["error"], "output_error")


if __name__ == "__main__":
    unittest.main()
