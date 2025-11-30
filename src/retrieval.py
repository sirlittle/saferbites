import pandas as pd
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

class SaferBitesEngine:
    def __init__(self, data_dir="data", use_reranker=True):
        self.data_dir = data_dir
        self.use_reranker = use_reranker
        
        # Load Data
        print("Loading data...")
        self.violations = pd.read_csv(f"{data_dir}/violations_processed.csv").fillna("")
        self.reviews = pd.read_csv(f"{data_dir}/reviews_processed.csv").fillna("")
        
        # Create BM25 indices
        print("Building BM25 indices...")
        self.viol_corpus = [self._tokenize(t) for t in self.violations["text"]]
        self.bm25_viol = BM25Okapi(self.viol_corpus)
        
        if not self.reviews.empty:
            self.rev_corpus = [self._tokenize(t) for t in self.reviews["text"]]
            self.bm25_rev = BM25Okapi(self.rev_corpus)
        else:
            self.bm25_rev = None
            
        # Load Reranker
        if self.use_reranker:
            print("Loading Reranker model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Engine initialized.")

    def _tokenize(self, text):
        return re.findall(r'\w+', str(text).lower())

    def search_bm25(self, query, top_k=30):
        tokenized_query = self._tokenize(query)
        
        results = []
        
        # Search Violations
        if not self.violations.empty:
            scores = self.bm25_viol.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:top_k]
            for idx in top_n:
                if scores[idx] > 0:
                    row = self.violations.iloc[idx]
                    results.append({
                        "doc_id": row["doc_id"],
                        "business_id": row["business_id"],
                        "business_name": row["business_name"],
                        "text": row["original_text"],
                        "score": scores[idx],
                        "source": "violation",
                        "tags": row["tags"]
                    })

        # Search Reviews
        if self.bm25_rev and not self.reviews.empty:
            scores = self.bm25_rev.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:top_k]
            for idx in top_n:
                if scores[idx] > 0:
                    row = self.reviews.iloc[idx]
                    results.append({
                        "doc_id": row["doc_id"],
                        "business_id": row["business_id"],
                        "business_name": row["business_name"],
                        "text": row["original_text"],
                        "score": scores[idx],
                        "source": "review",
                        "tags": row["tags"]
                    })
                    
        return results

    def rerank(self, query, results):
        if not results or not self.use_reranker:
            return results
            
        # Embed query
        query_emb = self.model.encode(query, convert_to_tensor=True)
        
        # Embed documents
        texts = [r["text"] for r in results]
        doc_embs = self.model.encode(texts, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
        
        # Update scores
        for i, r in enumerate(results):
            r["rerank_score"] = cosine_scores[i].item()
            
        # Sort by rerank score
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return results

    def aggregate_results(self, results, query):
        # Fusion Rule: score = (bm25 or rerank)
        
        keywords = self._tokenize(query)
        
        business_map = {}
        
        for r in results:
            bid = r["business_id"]
            if bid not in business_map:
                business_map[bid] = {
                    "business_id": bid,
                    "business_name": r["business_name"],
                    "total_score": 0,
                    "evidence": []
                }
            
            # Determine base score
            score = r.get("rerank_score", r["score"])
            
            # Apply category boost
            # Check if any tag in document matches query keywords
            boost = 1.0
            doc_tags = str(r["tags"]).split(",")
            
            # Logic to check if query keywords match document tags could be added here.
            # For now, we use a simple score summation.
            if r["tags"]:
                pass 
                
            fused_score = score
            
            business_map[bid]["total_score"] += fused_score
            business_map[bid]["evidence"].append(r)
            
        # Sort businesses by score
        ranked_businesses = sorted(business_map.values(), key=lambda x: x["total_score"], reverse=True)
        return ranked_businesses

if __name__ == "__main__":
    engine = SaferBitesEngine()
    q = "rat sighting"
    print(f"Query: {q}")
    res = engine.search_bm25(q)
    res = engine.rerank(q, res)
    agg = engine.aggregate_results(res, q)
    
    for b in agg[:3]:
        print(f"Business: {b['business_name']} ({b['business_id']}) - Score: {b['total_score']:.4f}")
        for e in b['evidence'][:2]:
            print(f"  - [{e['source']}] {e['text']} (Score: {e.get('rerank_score', e['score']):.4f})")
