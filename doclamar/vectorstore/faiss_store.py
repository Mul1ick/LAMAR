import os
import pickle
from typing import List

import faiss
import numpy as np

from doclamar.embeddings.hf_embedder import HFEmbedder
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class FaissSemanticStore:
    """
    Disk-backed FAISS semantic vector store.
    Embedding dimension is inferred automatically.
    """

    def __init__(self, store_path: str = "./doclamar/storage/faiss"):
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)

        self.index_file = os.path.join(self.store_path, "index.faiss")
        self.meta_file = os.path.join(self.store_path, "metadata.pkl")

        self.embedder = HFEmbedder()

        # 🔑 Infer embedding dimension safely
        test_embedding = self.embedder.embed(["dimension probe"])
        if hasattr(test_embedding, "cpu"):
            test_embedding = test_embedding.cpu().numpy()

        self.embedding_dim = test_embedding.shape[1]

        # Load or create FAISS index
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.metadata: List[ParsedChunkSchema] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []

    # -----------------------------
    # Indexing
    # -----------------------------

    def index_chunks(self, chunks: List[ParsedChunkSchema]):
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        else:
            embeddings = np.array(embeddings)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(chunks)

        self._persist()

    # -----------------------------
    # Retrieval
    # -----------------------------

    def search(self, query: str, top_k: int = 5) -> List[ParsedChunkSchema]:
        query_embedding = self.embedder.embed([query])

        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu().numpy()
        else:
            query_embedding = np.array(query_embedding)

        faiss.normalize_L2(query_embedding)

        _, indices = self.index.search(query_embedding, top_k)

        results: List[ParsedChunkSchema] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    # -----------------------------
    # Persistence
    # -----------------------------

    def _persist(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)
