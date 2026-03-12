from doclamar.embeddings.hf_embedder import HFEmbedder

embedder = HFEmbedder()

embeddings = embedder.embed([
    "machine learning algorithms",
    "neural networks and deep learning",
    "how to cook pasta"
])

print(len(embeddings))
print(embeddings[0].shape)
