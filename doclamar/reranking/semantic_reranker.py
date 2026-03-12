import numpy as np
from typing import List, Tuple

from doclamar.embeddings.hf_embedder import HFEmbedder
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class SemanticReRanker:
    def __init__(self):
        self.embedder = HFEmbedder()

    def rank(
        self,
        query: str,
        chunks: List[ParsedChunkSchema],
        top_k: int = 5,
    ) -> List[ParsedChunkSchema]:

        if not chunks:
            return []

        query_embedding = self.embedder.embed([query])[0]

        chunk_texts = [chunk.content for chunk in chunks]
        chunk_embeddings = self.embedder.embed(chunk_texts)

        scores: List[Tuple[float, ParsedChunkSchema]] = []

        for chunk, emb in zip(chunks, chunk_embeddings):
            score = float((query_embedding * emb).sum().item())

            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)

        ranked_chunks = [chunk for _, chunk in scores[:top_k]]
        return ranked_chunks
