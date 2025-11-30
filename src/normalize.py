import pandas as pd
import numpy as np
import re

DATA_DIR = "saferbites/data"

KEYWORDS = {
    "pests": ["rodent", "mouse", "mice", "roach", "insect", "rat", "fly", "flies", "vermin", "pest"],
    "temperature": ["cold", "hot", "temp", "thermometer", "degree", "cooling", "holding", "refrigerat"],
    "contamination": ["raw", "contam", "sanitize", "cross", "glove", "hand", "clean", "wash"]
}

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_tags(text):
    tags = []
    for category, words in KEYWORDS.items():
        if any(w in text for w in words):
            tags.append(category)
    return ",".join(tags)

def process_inspections():
    print("Processing Inspections...")
    df = pd.read_csv(f"{DATA_DIR}/inspections_raw.csv")
    
    # Keep relevant columns
    cols = ["camis", "dba", "violation_description", "street", "boro", "zipcode"]
    # Check which columns exist (case sensitive sometimes)
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].copy()
    
    if "violation_description" not in df.columns:
        print("Error: violation_description column not found.")
        return None

    df = df.dropna(subset=["violation_description"])
    
    # Normalize
    df["clean_text"] = df["violation_description"].apply(normalize_text)
    df["tags"] = df["clean_text"].apply(get_tags)
    
    # Create documents (one per violation)
    # We will rename columns to match a standard "document" schema
    # doc_id, business_id, text, source, tags, original_text, business_name
    
    documents = []
    for idx, row in df.iterrows():
        doc = {
            "doc_id": f"insp_{idx}",
            "business_id": row["camis"],
            "business_name": row.get("dba", "Unknown"),
            "text": row["clean_text"],
            "original_text": row["violation_description"],
            "source": "violation",
            "tags": row["tags"]
        }
        documents.append(doc)
        
    doc_df = pd.DataFrame(documents)
    doc_df.to_csv(f"{DATA_DIR}/violations_processed.csv", index=False)
    print(f"Saved {len(doc_df)} violation documents.")
    return df["camis"].unique()

def process_reviews(valid_business_ids):
    print("Processing Reviews...")
    try:
        df = pd.read_csv(f"{DATA_DIR}/reviews_raw.csv")
    except:
        print("No reviews file found.")
        return

    if "Review" not in df.columns:
        # Try 'text' or similar if schema differs
        if "text" in df.columns:
            df["Review"] = df["text"]
        else:
            print("Review column not found.")
            return

    documents = []
    count = 0
    
    # If we don't have many reviews, we can duplicate them to simulate volume, 
    # but for now just process what we have.
    
    for idx, row in df.iterrows():
        review_text = row["Review"]
        # Segment sentences
        sentences = re.split(r'[.!?]+', str(review_text))
        
        # Assign a random business ID
        if len(valid_business_ids) > 0:
            bid = np.random.choice(valid_business_ids)
        else:
            bid = "00000000"

        for s_idx, sentence in enumerate(sentences):
            clean_s = normalize_text(sentence).strip()
            if len(clean_s) < 5: continue # Skip too short
            
            tags = get_tags(clean_s)
            
            # Filter: Keep only if it has tags or food-safety-ish keyword
            if not tags: 
                continue
                
            doc = {
                "doc_id": f"rev_{idx}_{s_idx}",
                "business_id": bid,
                "business_name": "Unknown", # We don't know the name for random assignments
                "text": clean_s,
                "original_text": sentence.strip(),
                "source": "review",
                "tags": tags
            }
            documents.append(doc)
            count += 1
            
    doc_df = pd.DataFrame(documents)
    doc_df.to_csv(f"{DATA_DIR}/reviews_processed.csv", index=False)
    print(f"Saved {count} review sentence documents.")

if __name__ == "__main__":
    business_ids = process_inspections()
    if business_ids is not None:
        process_reviews(business_ids)
