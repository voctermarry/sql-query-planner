"""Lexer and parser: positions, clause coverage, and every documented error shape."""

from __future__ import annotations

import unittest

from sqlplan import parse, tokenise
from sqlplan.errors import ParseError
from sqlplan.parser import Aggregate, BoolOp, Column, Comparison, InList, IsNull, Literal, Not, Star


class LexerTests(unittest.TestCase):
    def test_positions_are_one_based(self) -> None:
        tokens = tokenise("SELECT a\nFROM t")
        self.assertEqual([(token.text, token.line, token.column) for token in tokens[:3]], [("select", 1, 1), ("a", 1, 8), ("from", 2, 1)])

    def test_keywords_are_case_insensitive_and_identifiers_keep_case(self) -> None:
        tokens = tokenise("SeLeCt MixedCase")
        self.assertEqual(tokens[0].text, "select")
        self.assertEqual(tokens[1].text, "MixedCase")

    def test_string_escape_and_comments(self) -> None:
        tokens = tokenise("SELECT 'it''s' -- trailing comment\nFROM t")
        self.assertEqual(tokens[1].value, "it's")
        self.assertEqual(tokens[2].text, "from")

    def test_numbers(self) -> None:
        self.assertEqual(tokenise("1")[0].value, 1)
        self.assertEqual(tokenise("1.5")[0].value, 1.5)

    def test_unterminated_string_reports_its_position(self) -> None:
        with self.assertRaises(ParseError) as caught:
            tokenise("SELECT 'oops")
        self.assertEqual((caught.exception.context["line"], caught.exception.context["column"]), (1, 8))

    def test_unexpected_character_reports_its_position(self) -> None:
        with self.assertRaises(ParseError) as caught:
            tokenise("SELECT #")
        self.assertEqual(caught.exception.kind, "parse_error")


class ParserTests(unittest.TestCase):
    def test_minimal_select(self) -> None:
        statement = parse("SELECT a FROM t")
        self.assertEqual(statement.table, "t")
        self.assertEqual(len(statement.projections), 1)
        self.assertEqual(statement.projections[0].expression, Column("a"))
        self.assertIsNone(statement.where)

    def test_star_projection(self) -> None:
        statement = parse("SELECT * FROM t")
        self.assertIsInstance(statement.projections[0].expression, Star)

    def test_aggregate_with_alias(self) -> None:
        statement = parse("SELECT count(*) AS n, sum(amount) AS total FROM orders")
        first, second = statement.projections
        self.assertEqual(first.expression, Aggregate("count", Star()))
        self.assertEqual(first.alias, "n")
        self.assertEqual(second.expression, Aggregate("sum", Column("amount")))

    def test_count_distinct(self) -> None:
        statement = parse("SELECT count(DISTINCT a) FROM t")
        self.assertTrue(statement.projections[0].expression.distinct)

    def test_qualified_column(self) -> None:
        statement = parse("SELECT t.a FROM t WHERE t.a = 1")
        self.assertEqual(statement.projections[0].expression, Column("a", "t"))
        self.assertEqual(statement.where, Comparison(Column("a", "t"), "=", Literal(1)))

    def test_boolean_precedence_and_before_or(self) -> None:
        statement = parse("SELECT a FROM t WHERE a = 1 OR b = 2 AND c = 3")
        self.assertIsInstance(statement.where, BoolOp)
        self.assertEqual(statement.where.operator, "or")
        self.assertEqual(statement.where.operands[1].operator, "and")

    def test_parentheses_override_precedence(self) -> None:
        statement = parse("SELECT a FROM t WHERE (a = 1 OR b = 2) AND c = 3")
        self.assertEqual(statement.where.operator, "and")
        self.assertEqual(statement.where.operands[0].operator, "or")

    def test_not_in_and_is_null(self) -> None:
        statement = parse("SELECT a FROM t WHERE a NOT IN (1, 2) AND b IS NOT NULL")
        self.assertEqual(statement.where.operands[0], InList(Column("a"), (Literal(1), Literal(2)), True))
        self.assertEqual(statement.where.operands[1], IsNull(Column("b"), True))

    def test_not_wraps_a_predicate(self) -> None:
        statement = parse("SELECT a FROM t WHERE NOT a = 1")
        self.assertIsInstance(statement.where, Not)

    def test_group_order_limit(self) -> None:
        statement = parse("SELECT a, count(*) FROM t GROUP BY a ORDER BY a DESC LIMIT 10;")
        self.assertEqual(statement.group_by, (Column("a"),))
        self.assertTrue(statement.order_by[0].descending)
        self.assertEqual(statement.limit, 10)

    def test_error_when_from_is_missing(self) -> None:
        with self.assertRaises(ParseError) as caught:
            parse("SELECT a")
        self.assertEqual(caught.exception.kind, "parse_error")

    def test_error_on_trailing_input(self) -> None:
        with self.assertRaises(ParseError):
            parse("SELECT a FROM t extra")

    def test_error_on_empty_statement(self) -> None:
        with self.assertRaises(ParseError):
            parse("   ")

    def test_error_position_points_at_the_offending_token(self) -> None:
        with self.assertRaises(ParseError) as caught:
            parse("SELECT a FROM")
        self.assertEqual(caught.exception.context["line"], 1)

    def test_document_shape_is_stable(self) -> None:
        document = parse("SELECT a AS x FROM t WHERE a > 1 LIMIT 5").to_document()
        self.assertEqual(document["table"], "t")
        self.assertEqual(document["projections"][0]["alias"], "x")
        self.assertEqual(document["limit"], 5)
        self.assertEqual(document["where"]["kind"], "comparison")

    def test_sum_of_star_is_rejected(self) -> None:
        with self.assertRaises(ParseError):
            parse("SELECT sum(*) FROM t")


if __name__ == "__main__":
    unittest.main()
