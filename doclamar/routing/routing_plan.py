from typing import List
from pydantic import BaseModel


class RoutingPlan(BaseModel):
    keywords: List[str]
    allowed_extensions: List[str]
    max_files: int
    use_content_preview: bool
