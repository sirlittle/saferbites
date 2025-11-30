import requests
import pandas as pd
import os
import io

DATA_DIR = "saferbites/data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_nyc_inspections():
    print("Downloading NYC DOHMH Restaurant Inspections (limit 5000)...")
    url = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv?$limit=5000&$order=inspection_date DESC"
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Save raw
        with open(f"{DATA_DIR}/inspections_raw.csv", "wb") as f:
            f.write(response.content)
        print("NYC Inspections downloaded.")
    except Exception as e:
        print(f"Error downloading NYC data: {e}")

def download_generic_reviews():
    print("Downloading Generic Restaurant Reviews...")
    urls = [
        "https://raw.githubusercontent.com/azratuni/Restaurant-Review-Scraping-and-Sentiment-Analysis/master/Restaurant%20reviews.csv",
        "https://raw.githubusercontent.com/chenzhivis/Analysis-and-Classification-of-Restaurant-Reviews/master/McDonalds-Yelp-Sentiment-DFE.csv",
        "https://raw.githubusercontent.com/futurexskill/ml-model-deployment/master/Restaurant_Reviews.tsv",
        "https://raw.githubusercontent.com/aaronkub/machine-learning-examples/master/restaurant-reviews-sentiment-analysis/Restaurant_Reviews.tsv"
    ]
    
    for url in urls:
        try:
            print(f"Trying {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                # Handle different formats
                if url.endswith('.tsv'):
                    df = pd.read_csv(io.StringIO(response.text), sep='\t')
                else:
                    df = pd.read_csv(io.StringIO(response.text))
                
                # Standardize column names
                if "Review Text" in df.columns:
                    df["Review"] = df["Review Text"]
                elif "review" in df.columns:
                    df["Review"] = df["review"]
                
                if "Review" in df.columns:
                    df.to_csv(f"{DATA_DIR}/reviews_raw.csv", index=False)
                    print(f"Generic Reviews downloaded from {url}.")
                    return
        except Exception as e:
            print(f"Failed: {e}")
            continue

    print("Could not download reviews. Creating dummy data.")
    data = {
        "Review": [
            "The food was great but I saw a rat.",
            "Amazing service and clean kitchen.",
            "Terrible hygiene, the waiter sneezed on my food.",
            "Food was cold and undercooked.",
            "Best pizza in town!",
            "Dirty tables and rude staff.",
            "I found a hair in my soup.",
            "Highly recommend this place.",
            "The chicken was raw.",
            "Roaches in the bathroom."
        ],
        "Liked": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{DATA_DIR}/reviews_raw.csv", index=False)

if __name__ == "__main__":
    download_nyc_inspections()
    download_generic_reviews()
