"""Exception hierarchy with stable `kind` values.

`ParseError` carries `line`/`column` because it is raised while reading user text; `PlanError` marks a
query that parses but cannot be planned (unknown column, aggregate without grouping key, ...).
"""

from __future__ import annotations


class SQLPlanError(Exception):
    """Base class for every error this package raises on purpose."""

    kind = "sql_plan_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = {key: value for key, value in context.items() if value is not None}

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error": self.kind, "message": self.message}
        document.update(self.context)
        return document


class ParseError(SQLPlanError):
    """Malformed input text: unexpected token, unbalanced parentheses, missing clause."""

    kind = "parse_error"

    def __init__(self, message: str, *, line: int | None = None, column: int | None = None, **context: object) -> None:
        super().__init__(message, line=line, column=column, **context)


class ValidationError(SQLPlanError):
    """A request that is well-formed but not allowed by this planner."""

    kind = "validation_error"


class PlanError(SQLPlanError):
    """The query parses but cannot be planned or executed."""

    kind = "plan_error"


class OutputError(SQLPlanError):
    """The output target cannot be used safely."""

    kind = "output_error"
