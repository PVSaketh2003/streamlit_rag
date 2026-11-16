"""Minimal FAISS vector store with LangChain-identical MMR retrieval.

Replaces `langchain_community.vectorstores.FAISS` so the serverless bundle stays small and the index can
live in Postgres (Vercel functions have no persistent disk). Behaviour matches LangChain exactly:

  * `IndexFlatL2` over the (already L2-normalised) embeddings, as `FAISS.from_documents` builds by default
  * `mmr_search` == `max_marginal_relevance_search_with_score_by_vector`: take the `fetch_k` nearest
    vectors by squared L2, re-rank them with `maximal_marginal_relevance`, return (row, distance) pairs

`tests.py` checks the results against langchain-community itself on random data.
"""
import numpy as np


def cosine_similarity(x, y):
    """Row-wise cosine similarity (same as langchain_community.utils.math.cosine_similarity)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.array([[]])
    xn = np.linalg.norm(x, axis=1)
    yn = np.linalg.norm(y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.dot(x, y.T) / np.outer(xn, yn)
    sim[np.isnan(sim) | np.isinf(sim)] = 0.0
    return sim


def maximal_marginal_relevance(query_embedding, embedding_list, lambda_mult=0.5, k=4):
    """Port of langchain_community.vectorstores.utils.maximal_marginal_relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    query_embedding = np.asarray(query_embedding)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class VectorIndex:
    """FAISS IndexFlatL2 + the chunk texts it indexes (row i == chunk i)."""

    def __init__(self, index, texts):
        if index.ntotal != len(texts):
            raise ValueError("The saved index and its stored chunks are out of sync. Re-index this video.")
        self.index = index
        self.texts = list(texts)

    @classmethod
    def build(cls, vectors, texts):
        import faiss
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2 or len(arr) != len(texts) or len(arr) == 0:
            raise ValueError("Embedding count does not match chunk count.")
        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr)
        return cls(index, texts)

    @property
    def ntotal(self):
        return self.index.ntotal

    @property
    def dim(self):
        return self.index.d

    def to_bytes(self):
        import faiss
        return faiss.serialize_index(self.index).tobytes()

    @classmethod
    def from_bytes(cls, blob, texts):
        import faiss
        index = faiss.deserialize_index(np.frombuffer(bytes(blob), dtype="uint8"))
        return cls(index, texts)

    def mmr_search(self, query_vector, k=4, fetch_k=20, lambda_mult=0.5):
        """[(row, squared_l2_distance)] — identical to LangChain's scored MMR search."""
        q = np.asarray([query_vector], dtype="float32")
        scores, indices = self.index.search(q, fetch_k)
        valid = [int(i) for i in indices[0] if i != -1]
        embeddings = [self.index.reconstruct(i) for i in valid]
        selected = maximal_marginal_relevance(q, embeddings, k=k, lambda_mult=lambda_mult)
        out = []
        for j in selected:
            if indices[0][j] == -1:
                continue
            out.append((int(indices[0][j]), float(scores[0][j])))
        return out
