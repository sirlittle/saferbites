import pandas as pd
import os
import re
from rank_bm25 import BM25Okapi
import glob

DATA_DIR = "saferbites/data"
LABELS_FILE = f"{DATA_DIR}/labeled_queries.csv"

# 50 Queries covering safety topics
QUERIES = [
    # Pests
    "rat sighting", "mice in kitchen", "roaches everywhere", "cockroach infestation", 
    "vermin problem", "fly infestation", "bugs in food", "rodent droppings", "mouse trap", "pests",
    # Temperature
    "cold food", "warm sushi", "refrigerator broken", "food not hot", "lukewarm chicken", 
    "freezer issue", "temperature violation", "improper holding temperature", "unsafe food temp", "cooling",
    # Contamination / Hygiene
    "hair in food", "dirty hands", "no gloves", "sneezing on food", "raw chicken", 
    "undercooked meat", "cross contamination", "dirty bathroom", "filthy kitchen", "moldy food",
    "unwashed produce", "bare hand contact", "sanitizer missing", "soap empty", "dirty plates",
    # General
    "health hazard", "sanitary violation", "grade pending", "shut down by health department", "food poisoning",
    "sick waiter", "coughing staff", "hygiene issues", "dirty floors", "garbage piling up",
    "bad smell", "sewage odor", "plumbing issues", "water leak", "no hot water"
]

def tokenize(text):
    return re.findall(r'\w+', str(text).lower())

def generate_ground_truth():
    print("Generating 'Silver Standard' Ground Truth from Real Data...")
    
    # Load processed documents
    try:
        viol_df = pd.read_csv(f"{DATA_DIR}/violations_processed.csv").fillna("")
        rev_df = pd.read_csv(f"{DATA_DIR}/reviews_processed.csv").fillna("")
        all_docs = pd.concat([viol_df, rev_df], ignore_index=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Scanning {len(all_docs)} documents for query matches...")
    
    ground_truth = []
    
    # Build a simple inverted index for faster lookup
    # Word -> Set of business_ids
    index = {}
    for _, row in all_docs.iterrows():
        bid = str(row["business_id"])
        tokens = tokenize(row["text"])
        for t in tokens:
            if t not in index:
                index[t] = set()
            index[t].add(bid)
            
    for query in QUERIES:
        q_tokens = tokenize(query)
        relevant_bids = None
        
        # Boolean AND
        for t in q_tokens:
            if t in index:
                if relevant_bids is None:
                    relevant_bids = index[t].copy()
                else:
                    relevant_bids.intersection_update(index[t])
            else:
                # If a query term is missing from corpus, result is empty for AND
                relevant_bids = set()
                break
        
        # If empty, try Boolean OR (relaxed) for "Partial Match" relevance
        # This prevents having 0 results for tough queries
        if not relevant_bids:
             relevant_bids = set()
             for t in q_tokens:
                 if t in index:
                     relevant_bids.update(index[t])
        
        if relevant_bids:
            ground_truth.append({
                "query": query,
                "relevant_business_ids": " ".join(list(relevant_bids)[:50]) # Limit to 50
            })

    gt_df = pd.DataFrame(ground_truth)
    gt_df.to_csv(LABELS_FILE, index=False)
    print(f"Generated labels for {len(gt_df)} queries. Saved to {LABELS_FILE}.")

if __name__ == "__main__":
    generate_ground_truth()
