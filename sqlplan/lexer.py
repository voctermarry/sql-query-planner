"""Tokeniser for the SQL subset this planner accepts.

Positions are 1-based and travel with every token, so a parse error can always say where it happened.
Keywords are case-insensitive; identifiers keep their original spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ParseError

KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "asc",
    "desc",
    "limit",
    "and",
    "or",
    "not",
    "is",
    "null",
    "in",
    "as",
    # join syntax: FROM a INNER JOIN b ON a.k = b.k
    "inner",
    "join",
    "on",
    # 'distinct' must be a keyword: without it `count(DISTINCT a)` lexed DISTINCT as a column name and
    # the parser then demanded ')' while 'a' was still unconsumed (caught by test_count_distinct).
    "distinct",
)
AGGREGATES = ("count", "sum", "min", "max", "avg")
OPERATORS = ("<>", "<=", ">=", "=", "<", ">", "*", ",", "(", ")", "+", "-", ".", ";")

IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
# NOTE: '.' is deliberately absent. It was in this set first, which made `t.a` lex as ONE identifier
# named "t.a" and silently disabled every qualified column (the parser's `.` operator never matched).
# Number literals consume their own '.' in the DIGITS branch, so nothing else needs it here.
IDENT_BODY = IDENT_START | set("0123456789")
DIGITS = set("0123456789")


@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # keyword | identifier | number | string | operator | eof
    text: str
    line: int
    column: int
    value: object | None = None

    def to_document(self) -> dict[str, object]:
        document = {"kind": self.kind, "text": self.text, "line": self.line, "column": self.column}
        if self.value is not None:
            document["value"] = self.value
        return document


def tokenise(text: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    line = 1
    column = 1
    length = len(text)

    def advance(count: int = 1) -> None:
        nonlocal index, line, column
        for _ in range(count):
            if index < length and text[index] == "\n":
                line += 1
                column = 1
            else:
                column += 1
            index += 1

    while index < length:
        character = text[index]
        if character in " \t\r\n":
            advance()
            continue
        if character == "-" and text.startswith("--", index):
            while index < length and text[index] != "\n":
                advance()
            continue
        if character == "'":
            start_line, start_column = line, column
            advance()
            chunk: list[str] = []
            while True:
                if index >= length:
                    raise ParseError("unterminated string literal", line=start_line, column=start_column)
                if text[index] == "'":
                    if index + 1 < length and text[index + 1] == "'":
                        chunk.append("'")
                        advance(2)
                        continue
                    advance()
                    break
                chunk.append(text[index])
                advance()
            tokens.append(Token("string", "".join(chunk), start_line, start_column, "".join(chunk)))
            continue
        if character in DIGITS:
            start_line, start_column = line, column
            start = index
            while index < length and (text[index] in DIGITS or text[index] == "."):
                advance()
            raw = text[start:index]
            try:
                value: object = float(raw) if "." in raw else int(raw)
            except ValueError as error:
                raise ParseError(f"invalid number: {raw}", line=start_line, column=start_column) from error
            tokens.append(Token("number", raw, start_line, start_column, value))
            continue
        if character in IDENT_START:
            start_line, start_column = line, column
            start = index
            while index < length and text[index] in IDENT_BODY:
                advance()
            raw = text[start:index]
            lowered = raw.lower()
            if lowered in KEYWORDS or lowered in AGGREGATES:
                tokens.append(Token("keyword", lowered, start_line, start_column, lowered))
            else:
                tokens.append(Token("identifier", raw, start_line, start_column, raw))
            continue
        for operator in OPERATORS:
            if text.startswith(operator, index):
                tokens.append(Token("operator", operator, line, column, operator))
                advance(len(operator))
                break
        else:
            raise ParseError(f"unexpected character: {character!r}", line=line, column=column)
    tokens.append(Token("eof", "", line, column, None))
    return tokens
