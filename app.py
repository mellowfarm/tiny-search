from flask import Flask, request, render_template, jsonify
import faiss, pickle
from sentence_transformers import SentenceTransformer, CrossEncoder

index = faiss.read_index("index.faiss")
with open("passages.pkl", "rb") as f:
    passages = pickle.load(f)

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# search fxn, takes a query and returns the top 5 most relevant passages
def search(query, top_k):
    # layer 1: bi-encoder search to get top k passages
    query_embedding = bi_encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    bi_scores, indices = index.search(query_embedding, top_k) # search the index for the top k most similar passages
    
    # layer 2: cross-encoder re-ranking to get the top 5 passages
    cross_inputs = [[query, passages[i]["text"]] for i in indices[0]]
    cross_scores = cross_encoder.predict(cross_inputs)
    top_indices = sorted(range(len(cross_scores)), key=lambda i: cross_scores[i], reverse=True)[:5] # get indices of top 5 passages
    return [passages[indices[0][i]] for i in top_indices]

app = Flask(__name__)

@app.route("/") # home page route
def home():
    return render_template("index.html", query = None, results = []) # renders the index.html template

@app.route("/search")
def search_page():
    query = request.args.get("q", "")
    results = search(query, top_k=10)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)