from pydantic import BaseModel


class Step(BaseModel):
    n: int
    element: str
    note: str = ""


class ContextInfo(BaseModel):
    h: str
    pulled: list[str] = []
    active_count: int = 0


class FinalAnswer(BaseModel):
    answer: str
    steps: list[Step]
    context: ContextInfo
