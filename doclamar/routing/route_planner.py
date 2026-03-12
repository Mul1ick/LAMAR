import json
from doclamar.llm.llm_provider import get_llm
from doclamar.routing.routing_plan import RoutingPlan
from doclamar.utils.json_utils import extract_json


class RoutePlanner:
    def __init__(self):
        self.llm = get_llm()

    def plan(self, query: str) -> RoutingPlan:
        """
        Uses an LLM to decide a routing strategy.
        Falls back to a safe deterministic plan if JSON extraction fails.
        """

        prompt = f"""
You are a planning module.

Your task is to output a JSON object ONLY.
Do NOT include explanations, markdown, or extra text.

If you include anything outside JSON, the system will fail.

User query:
{query}

Return EXACTLY one JSON object in this format:

{{
  "keywords": ["machine learning", "model", "data"],
  "allowed_extensions": ["pdf", "txt"],
  "max_files": 30,
  "use_content_preview": true
}}

REMEMBER:
- Output JSON only
- No extra text
"""

        response = self.llm.generate(prompt)

        # Try strict JSON extraction
        try:
            data = extract_json(response)
            return RoutingPlan(**data)

        except Exception:
            # Safe fallback plan (robust default)
            # Remove common stopwords for routing
            stopwords = {
            '''    "explain", "tell", "me", "about", "something",
                "what", "is", "the", "a", '''"an"
            }

            keywords = [
                word for word in query.lower().split()
                if word not in stopwords and len(word) > 2
            ]
            return RoutingPlan(
                keywords=query.lower().split(),
                allowed_extensions=["pdf", "txt"],
                max_files=20,
                use_content_preview=True
            )
