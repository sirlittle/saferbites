# SaferBites 🍽️🔍

A retrieval system that fuses official health-inspection violations with public review text to answer safety queries with evidence.

## Setup

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    ```bash
    python3 src/download_data.py
    ```
    This downloads NYC Inspection data and a sample reviews dataset.

3.  **Normalize Data**:
    ```bash
    python3 src/normalize.py
    ```
    This processes the raw CSVs into searchable documents and tags them (pests, temperature, etc.).

## Running the Search Engine (UI)

1.  Start the Flask app:
    ```bash
    python3 ui/app.py
    ```
2.  Open your browser at `http://127.0.0.1:5000`.
3.  Search for queries like "rats", "roaches", "cold food".

## Evaluation

To compute MAP/NDCG metrics on a sample query set:

```bash
python3 src/evaluate.py
```

## Project Structure

- `data/`: Raw and processed data.
- `src/`: Core logic.
    - `download_data.py`: Data fetching.
    - `normalize.py`: Cleaning and tagging.
    - `retrieval.py`: BM25 + Reranker engine.
    - `evaluate.py`: Metrics calculation.
- `ui/`: Web interface.
    - `app.py`: Flask application.
