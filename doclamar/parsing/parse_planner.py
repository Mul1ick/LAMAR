from doclamar.llm.llm_provider import get_llm
from doclamar.parsing.parsing_plan import ParsingPlan
from doclamar.schemas.document import DocumentSchema
from doclamar.utils.json_utils import extract_json
import json


class ParsePlanner:
    def __init__(self):
        self.llm = get_llm()

    def plan(self, query: str, document):
        """
        Decide how much of the document to parse.
        Always returns a valid ParsingPlan.
        """

        prompt = f"""
You are deciding how to parse a document.

User query:
{query}

Document:
{document.file_name} ({document.file_type}, {document.size_kb} KB)

Decide whether to parse the full document or only a part.

Return JSON with:
- parse_full (boolean)
- max_chars (integer)
- reason (string)
"""

        try:
            response = self.llm.generate(prompt)
            data = json.loads(response)
        except Exception:
            # Safe fallback
            data = {
                "parse_full": True,
                "max_chars": 3000,
                "reason": "Fallback parsing decision due to planner failure."
            }

        # 🔒 GUARANTEE required fields
        data.setdefault(
            "reason",
            "Parsing strategy selected based on query-document relevance."
        )

        data.setdefault("max_chars", 3000)
        data.setdefault("parse_full", True)

        return ParsingPlan(**data)
