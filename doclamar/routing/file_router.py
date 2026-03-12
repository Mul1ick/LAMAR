import os
from typing import List

from doclamar.schemas.document import DocumentSchema
from doclamar.routing.routing_plan import RoutingPlan


class FileRouter:
    def route(
        self,
        root_path: str,
        plan: RoutingPlan,
    ) -> List[DocumentSchema]:

        results: List[DocumentSchema] = []

        for root, _, files in os.walk(root_path):
            for file in files:
                if len(results) >= plan.max_files:
                    return results

                ext = file.split(".")[-1].lower()
                if ext not in plan.allowed_extensions:
                    continue

                full_path = os.path.join(root, file)

                # --- Stage 1: filename/path filtering ---
                normalized_name = (
                                file.lower()
                                .replace("_", " ")
                                .replace("-", " ")
                )

                searchable_text = f"{full_path.lower()} {normalized_name}"

                #searchable_text = f"{full_path.lower()} {file.lower()}"
                keyword_match = any(
                    k.lower() in searchable_text for k in plan.keywords
                )

                # --- Stage 2: optional content preview ---
                if not keyword_match and plan.use_content_preview:
                    try:
                        with open(full_path, "rb") as f:
                            preview = f.read(2000).decode(
                                errors="ignore"
                            ).lower()
                        keyword_match = any(
                            k.lower() in preview for k in plan.keywords
                        )
                    except Exception:
                        pass  # safely ignore unreadable files

                if not keyword_match:
                    continue

                size_kb = round(os.path.getsize(full_path) / 1024, 2)

                results.append(
                    DocumentSchema(
                        file_path=full_path,
                        file_name=file,
                        file_type=ext,
                        size_kb=size_kb,
                    )
                )

        return results
