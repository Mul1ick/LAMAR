import os
import hashlib
from typing import List

from openviking.storage import VectorStore

from doclamar.embeddings.hf_embedder import HFEmbedder
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class OpenVikingIndexer:
    """
    Handles persistent semantic indexing of parsed document chunks using OpenViking.
    """

    def __init__(self, index_path: str = "./doclamar/storage/viking"):
        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)

        self.embedder = HFEmbedder()

        self.store = VectorStore(
            persist_dir=self.index_path,
            embedding_function=self.embed_text,
        )

    def embed_text(self, texts: List[str]):
        return self.embedder.embed(texts)

    def _chunk_id(self, chunk: ParsedChunkSchema) -> str:
        base = (
            chunk.metadata.get("file_name", "")
            + str(chunk.metadata.get("chunk_index", ""))
            + chunk.content[:100]
        )
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def index_chunks(self, chunks: List[ParsedChunkSchema]):
        if not chunks:
            return

        texts = []
        metadatas = []
        ids = []

        for chunk in chunks:
            texts.append(chunk.content)
            metadatas.append(chunk.metadata)
            ids.append(self._chunk_id(chunk))

        self.store.add(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        self.store.persist()
