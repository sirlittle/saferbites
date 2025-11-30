import pandas as pd
import numpy as np
from retrieval import SaferBitesEngine
from sklearn.metrics import ndcg_score
import os

def compute_metrics(retrieved_ids, relevant_ids, k=10):
    # retrieved_ids: list of business_ids returned
    # relevant_ids: set of relevant business_ids
    
    retrieved_ids = retrieved_ids[:k]
    
    # Precision @ K
    relevant_count = sum(1 for bid in retrieved_ids if bid in relevant_ids)
    precision = relevant_count / k if k > 0 else 0
    
    # Recall @ K (Recall is usually over total relevant, but here we just measure relative recall)
    recall = relevant_count / len(relevant_ids) if len(relevant_ids) > 0 else 0
    
    # NDCG @ K
    # y_true: 1 if relevant else 0
    y_true = [1 if bid in relevant_ids else 0 for bid in retrieved_ids]
    # If we returned fewer than k, pad with 0
    if len(y_true) < k:
        y_true += [0] * (k - len(y_true))
    
    # y_score: rank score (simplest is just reverse rank or predicted score)
    # ideally we use the actual scores, but here we assume the list is ranked
    y_score = [k - i for i in range(len(y_true))]
    
    # ndcg_score requires shape (n_samples, n_labels)
    # We treat this single query as 1 sample
    try:
        ndcg = ndcg_score([y_true], [y_score], k=k)
    except:
        ndcg = 0.0
        
    return precision, recall, ndcg

def main():
    engine = SaferBitesEngine()
    
    labels_file = "saferbites/data/labeled_queries.csv"
    if not os.path.exists(labels_file):
        print("Labels file not found. Generating dummy labels for testing...")
        # Generate some dummy ground truth
        # We will search for keywords and mark the top 5 results as "relevant" for this test
        queries = ["mice", "roach", "dirty", "cold food", "glove"]
        with open(labels_file, "w") as f:
            f.write("query,relevant_business_ids\n")
            for q in queries:
                res = engine.search_bm25(q) # Use BM25 to find candidates
                top_ids = [str(r["business_id"]) for r in res[:5]]
                ids_str = " ".join(top_ids)
                f.write(f'{q},"{ids_str}"\n')
    
    df = pd.read_csv(labels_file)
    
    metrics = {"p@10": [], "ndcg@10": []}
    
    print(f"Evaluating on {len(df)} queries...")
    
    for idx, row in df.iterrows():
        query = row["query"]
        rel_ids = set(str(x) for x in str(row["relevant_business_ids"]).split())
        
        # Run System
        # 1. BM25
        res = engine.search_bm25(query)
        # 2. Rerank
        res = engine.rerank(query, res)
        # 3. Aggregate
        agg_res = engine.aggregate_results(res, query)
        
        retrieved_ids = [str(b["business_id"]) for b in agg_res]
        
        p10, r10, ndcg = compute_metrics(retrieved_ids, rel_ids, k=10)
        metrics["p@10"].append(p10)
        metrics["ndcg@10"].append(ndcg)
        
        print(f"Q: {query} -> P@10: {p10:.2f}, NDCG@10: {ndcg:.2f}")

    print("-" * 30)
    print(f"Mean P@10: {np.mean(metrics['p@10']):.4f}")
    print(f"Mean NDCG@10: {np.mean(metrics['ndcg@10']):.4f}")

if __name__ == "__main__":
    main()
