from pydantic import BaseModel


class ParsingPlan(BaseModel):
    parse_full: bool
    max_chars: int
    reason: str
