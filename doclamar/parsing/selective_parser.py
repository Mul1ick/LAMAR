from typing import List
import PyPDF2

from doclamar.schemas.document import DocumentSchema
from doclamar.schemas.parsed_chunk import ParsedChunkSchema
from doclamar.parsing.parsing_plan import ParsingPlan


class SelectiveParser:
    def parse(
        self,
        document: DocumentSchema,
        plan: ParsingPlan,
    ) -> List[ParsedChunkSchema]:

        text = ""

        if document.file_type == "pdf":
            with open(document.file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
                    if not plan.parse_full and len(text) >= plan.max_chars:
                        break

        elif document.file_type == "txt":
            with open(document.file_path, "r", encoding="utf-8", errors="ignore") as f:
                if plan.parse_full:
                    text = f.read()
                else:
                    text = f.read(plan.max_chars)

        if not text:
            return []

        return [
            ParsedChunkSchema(
                content=text,
                metadata={
                    "file_path": document.file_path,
                    "file_name": document.file_name,
                    "parse_reason": plan.reason,
                },
            )
        ]
