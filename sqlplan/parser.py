"""Recursive-descent parser for the SELECT subset, producing a small immutable AST.

Grammar (everything is optional except SELECT ... FROM ...):

    select   := SELECT projections FROM table [WHERE cond]
                [GROUP BY column (',' column)*] [ORDER BY column [ASC|DESC]] [LIMIT number] [';']
    proj     := expr [AS identifier] | '*'
    expr     := aggregate '(' (column | '*') ')' | column | literal
    cond     := or_expr
    or_expr  := and_expr (OR and_expr)*
    and_expr := primary (AND primary)*
    primary  := NOT primary | '(' cond ')' | atom [NOT] IN '(' literal (',' literal)* ')'
                | atom IS [NOT] NULL | atom ('='|'<>'|'<'|'<='|'>'|'>=') atom
    atom     := column | literal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .errors import ParseError, ValidationError
from .lexer import AGGREGATES, Token, tokenise


# -- AST -----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Literal:
    value: object


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    table: str | None = None


@dataclass(frozen=True, slots=True)
class Star:
    pass


@dataclass(frozen=True, slots=True)
class Aggregate:
    function: str
    argument: Column | Star | Literal
    distinct: bool = False


@dataclass(frozen=True, slots=True)
class Comparison:
    left: object
    operator: str
    right: object


@dataclass(frozen=True, slots=True)
class InList:
    operand: object
    values: tuple[object, ...]
    negated: bool = False


@dataclass(frozen=True, slots=True)
class IsNull:
    operand: object
    negated: bool = False


@dataclass(frozen=True, slots=True)
class Not:
    operand: object


@dataclass(frozen=True, slots=True)
class BoolOp:
    operator: str  # and | or
    operands: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class OrderKey:
    column: Column
    descending: bool = False


@dataclass(frozen=True, slots=True)
class Projection:
    expression: object
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class Select:
    projections: tuple[Projection, ...]
    table: str
    where: object | None = None
    group_by: tuple[Column, ...] = ()
    order_by: tuple[OrderKey, ...] = ()
    limit: int | None = None

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "projections": [_projection(item) for item in self.projections],
            "table": self.table,
        }
        if self.where is not None:
            document["where"] = _expression(self.where)
        if self.group_by:
            document["groupBy"] = [item.name for item in self.group_by]
        if self.order_by:
            document["orderBy"] = [{"column": key.column.name, "descending": key.descending} for key in self.order_by]
        if self.limit is not None:
            document["limit"] = self.limit
        return document


def _projection(projection: Projection) -> dict[str, object]:
    document = {"expression": _expression(projection.expression)}
    if projection.alias:
        document["alias"] = projection.alias
    return document


def _expression(node: object) -> dict[str, object]:
    if isinstance(node, Literal):
        return {"kind": "literal", "value": node.value}
    if isinstance(node, Column):
        return {"kind": "column", "name": node.name} | ({"table": node.table} if node.table else {})
    if isinstance(node, Star):
        return {"kind": "star"}
    if isinstance(node, Aggregate):
        return {"kind": "aggregate", "function": node.function, "argument": _expression(node.argument), "distinct": node.distinct}
    if isinstance(node, Comparison):
        return {"kind": "comparison", "operator": node.operator, "left": _expression(node.left), "right": _expression(node.right)}
    if isinstance(node, InList):
        return {"kind": "in", "negated": node.negated, "operand": _expression(node.operand), "values": [_expression(value) for value in node.values]}
    if isinstance(node, IsNull):
        return {"kind": "isNull", "negated": node.negated, "operand": _expression(node.operand)}
    if isinstance(node, Not):
        return {"kind": "not", "operand": _expression(node.operand)}
    if isinstance(node, BoolOp):
        return {"kind": node.operator, "operands": [_expression(operand) for operand in node.operands]}
    raise ValidationError(f"cannot render expression: {type(node).__name__}")


# -- parser --------------------------------------------------------------------------------------
class Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.index = 0

    # -- helpers ---------------------------------------------------------------------------------
    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        if token.kind != "eof":
            self.index += 1
        return token

    def fail(self, message: str, token: Token | None = None) -> ParseError:
        where = token or self.current
        return ParseError(message, line=where.line, column=where.column, found=where.text)

    def eat_keyword(self, keyword: str) -> Token | None:
        if self.current.kind == "keyword" and self.current.text == keyword:
            return self.advance()
        return None

    def expect_keyword(self, keyword: str) -> Token:
        token = self.eat_keyword(keyword)
        if token is None:
            raise self.fail(f"expected {keyword.upper()}")
        return token

    def eat_operator(self, operator: str) -> Token | None:
        if self.current.kind == "operator" and self.current.text == operator:
            return self.advance()
        return None

    def expect_operator(self, operator: str) -> Token:
        token = self.eat_operator(operator)
        if token is None:
            raise self.fail(f"expected {operator!r}")
        return token

    def expect_identifier(self) -> str:
        if self.current.kind == "identifier":
            return str(self.advance().text)
        raise self.fail("expected an identifier")

    # -- statement -------------------------------------------------------------------------------
    def parse_statement(self) -> Select:
        self.expect_keyword("select")
        projections = self.parse_projections()
        self.expect_keyword("from")
        table = self.expect_identifier()
        where = None
        group_by: tuple[Column, ...] = ()
        order_by: tuple[OrderKey, ...] = ()
        limit: int | None = None
        if self.eat_keyword("where"):
            where = self.parse_condition()
        if self.eat_keyword("group"):
            self.expect_keyword("by")
            group_by = tuple(self.parse_column_group())
        if self.eat_keyword("order"):
            self.expect_keyword("by")
            order_by = tuple(self.parse_order_keys())
        if self.eat_keyword("limit"):
            limit = self.parse_limit()
        self.eat_operator(";")
        if self.current.kind != "eof":
            raise self.fail("unexpected trailing input")
        if not projections:
            raise self.fail("SELECT needs at least one projection", self.tokens[0])
        return Select(
            projections=projections,
            table=table,
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
        )

    def parse_projections(self) -> tuple[Projection, ...]:
        items = [self.parse_projection()]
        while self.eat_operator(","):
            items.append(self.parse_projection())
        return tuple(items)

    def parse_projection(self) -> Projection:
        if self.eat_operator("*"):
            return Projection(Star())
        expression = self.parse_value()
        alias = None
        if self.eat_keyword("as"):
            alias = self.expect_identifier()
        else:
            alias = None
        return Projection(expression, alias)

    def parse_column_group(self) -> list[Column]:
        columns = [self.parse_column()]
        while self.eat_operator(","):
            columns.append(self.parse_column())
        return columns

    def parse_order_keys(self) -> list[OrderKey]:
        keys = [self.parse_order_key()]
        while self.eat_operator(","):
            keys.append(self.parse_order_key())
        return keys

    def parse_order_key(self) -> OrderKey:
        column = self.parse_column()
        descending = False
        if self.eat_keyword("desc"):
            descending = True
        else:
            self.eat_keyword("asc")
        return OrderKey(column, descending)

    def parse_limit(self) -> int:
        if self.current.kind != "number" or not isinstance(self.current.value, int):
            raise self.fail("LIMIT expects a non-negative integer")
        value = int(self.advance().value)  # type: ignore[arg-type]
        if value < 0:
            raise self.fail("LIMIT must be non-negative")
        return value

    def parse_column(self) -> Column:
        name = self.expect_identifier()
        table = None
        if self.eat_operator("."):
            table, name = name, self.expect_identifier()
        return Column(name=name, table=table)

    # -- values ----------------------------------------------------------------------------------
    def parse_value(self) -> object:
        if self.current.kind == "keyword" and self.current.text in AGGREGATES:
            return self.parse_aggregate()
        return self.parse_atom()

    def parse_aggregate(self) -> Aggregate:
        function = str(self.advance().text)
        distinct = False
        self.expect_operator("(")
        if self.eat_keyword("distinct"):
            distinct = True
        if self.eat_operator("*"):
            argument: object = Star()
        elif self.current.kind == "number" and isinstance(self.current.value, int) and not distinct:
            argument = Literal(int(self.advance().value))  # type: ignore[arg-type]
        else:
            argument = self.parse_column()
        self.expect_operator(")")
        if function != "count" and isinstance(argument, Star):
            raise self.fail(f"{function.upper()} does not accept '*'")
        return Aggregate(function=function, argument=argument, distinct=distinct)

    def parse_atom(self) -> object:
        token = self.current
        if token.kind == "number":
            self.advance()
            return Literal(token.value)
        if token.kind == "string":
            self.advance()
            return Literal(token.value)
        if token.kind == "identifier":
            return self.parse_column()
        if self.eat_operator("("):
            inner = self.parse_value()
            self.expect_operator(")")
            return inner
        raise self.fail("expected a value")

    # -- conditions ------------------------------------------------------------------------------
    def parse_condition(self) -> object:
        return self.parse_or()

    def parse_or(self) -> object:
        operands = [self.parse_and()]
        while self.eat_keyword("or"):
            operands.append(self.parse_and())
        return operands[0] if len(operands) == 1 else BoolOp("or", tuple(operands))

    def parse_and(self) -> object:
        operands = [self.parse_not()]
        while self.eat_keyword("and"):
            operands.append(self.parse_not())
        return operands[0] if len(operands) == 1 else BoolOp("and", tuple(operands))

    def parse_not(self) -> object:
        if self.eat_keyword("not"):
            return Not(self.parse_not())
        if self.eat_operator("("):
            inner = self.parse_condition()
            self.expect_operator(")")
            return inner
        return self.parse_predicate()

    def parse_predicate(self) -> object:
        operand = self.parse_atom()
        if self.eat_keyword("is"):
            negated = bool(self.eat_keyword("not"))
            self.expect_keyword("null")
            return IsNull(operand, negated)
        negated = bool(self.eat_keyword("not"))
        if self.eat_keyword("in"):
            self.expect_operator("(")
            values = [self.parse_atom()]
            while self.eat_operator(","):
                values.append(self.parse_atom())
            self.expect_operator(")")
            return InList(operand, tuple(values), negated)
        if negated:
            raise self.fail("expected IN after NOT")
        for operator in ("=", "<>", "<", "<=", ">", ">="):
            if self.eat_operator(operator):
                return Comparison(operand, operator, self.parse_atom())
        raise self.fail("expected a comparison operator")


def parse(text: str) -> Select:
    """Parse one SELECT statement."""
    stripped = text.strip()
    if not stripped:
        raise ParseError("empty statement", line=1, column=1)
    return Parser(tokenise(stripped)).parse_statement()
