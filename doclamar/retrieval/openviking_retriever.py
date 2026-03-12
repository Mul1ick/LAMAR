from typing import List

from openviking.store import VectorStore

from doclamar.embeddings.hf_embedder import HFEmbedder
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class OpenVikingRetriever:
    """
    Semantic retriever backed by OpenViking.
    """

    def __init__(self, index_path: str = "./doclamar/storage/viking"):
        self.index_path = index_path

        self.embedder = HFEmbedder()

        self.store = VectorStore(
            persist_dir=self.index_path,
            embedding_function=self.embed_text,
        )

    def embed_text(self, texts: List[str]):
        return self.embedder.embed(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[ParsedChunkSchema]:

        results = self.store.search(
            query=query,
            k=top_k,
        )

        chunks: List[ParsedChunkSchema] = []

        for res in results:
            chunks.append(
                ParsedChunkSchema(
                    content=res["text"],
                    metadata=res.get("metadata", {}),
                )
            )

        return chunks
