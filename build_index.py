
import pickle
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

passages = []
for article in ds:
    words = article["text"].split()
    for i in range(0, len(words), 200):
        chunk = " ".join(words[i:i+200])
        passages.append({"title": article["title"], "text": chunk})


print("\nEncoding passages with bi-encoder (this takes a while)...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = bi_encoder.encode(
    [p["text"] for p in passages],
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print("\nBuilding FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index contains {index.ntotal:,} vectors")

print("\nSaving index.faiss and passages.pkl...")
faiss.write_index(index, "index.faiss")
with open("passages.pkl", "wb") as f:
    pickle.dump(passages, f)

print("\nDone! You can now run: python app.py")