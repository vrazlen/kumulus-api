from pydantic import BaseModel
from datetime import datetime

class Settlement(BaseModel):
    id: int
    name: str
    kumuh_score: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True