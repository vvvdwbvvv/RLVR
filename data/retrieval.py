import json

import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index("index.faiss")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


class Retrieve:
    def __init__(self):
        self.top_k = 5

    def search(self, query: str, top_k: int):
        q = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idx = index.search(q, top_k)

        results = []
        for rank, score in enumerate(scores[0], start=1):
            results.append(
                {"rank": rank, "score": float(score), "query": query.strip()}
            )
        return results


if __name__ == "__main__":
    query = """Epigastric fullness. Try H2 blockers and mopride. PES if no improved.
    1100318 Mild improved, arrange PES.
    1100327 PES showed RE (A), 1st course of PPI. Keep mopride for fullness. Bed side echo: mild FL, GB polyps up to 0.6 cm."""
    print(json.dumps(Retrieve.search(query, top_k=5), indent=2, ensure_ascii=False))
