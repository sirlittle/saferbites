from flask import Flask, request, render_template_string
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval import SaferBitesEngine

app = Flask(__name__)

# Initialize Engine once
print("Initializing SaferBites Engine...")
engine = SaferBitesEngine()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SaferBites</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .evidence-box { background-color: #f8f9fa; border-left: 4px solid #dc3545; padding: 10px; margin-top: 5px; }
        .highlight { background-color: #ffeeba; font-weight: bold; }
        .score { font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body class="container py-5">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <h1 class="text-center mb-4">SaferBites 🍽️🔍</h1>
            <p class="text-center text-muted">Find safe places to eat. Search for "rats", "dirty", "cold food"...</p>
            
            <form action="/search" method="get" class="d-flex gap-2 mb-5">
                <input type="text" name="q" class="form-control form-control-lg" placeholder="e.g. rodents in manhattan" value="{{ query }}">
                <button type="submit" class="btn btn-primary btn-lg">Search</button>
            </form>

            {% if query %}
                <h3>Results for "{{ query }}"</h3>
                <hr>
                {% if results %}
                    {% for business in results %}
                        <div class="card mb-3">
                            <div class="card-body">
                                <h4 class="card-title">{{ business.business_name }} 
                                    <span class="score float-end">Score: {{ "%.2f"|format(business.total_score) }}</span>
                                </h4>
                                <h6 class="card-subtitle mb-2 text-muted">ID: {{ business.business_id }}</h6>
                                
                                <div class="mt-3">
                                    <h6>Evidence:</h6>
                                    {% for ev in business.evidence[:3] %}
                                        <div class="evidence-box">
                                            <small class="text-uppercase fw-bold text-{{ 'danger' if ev.source=='violation' else 'warning' }}">{{ ev.source }}</small>
                                            <p class="mb-0">{{ ev.text }}</p>
                                        </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No results found.</p>
                {% endif %}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, query="", results=[])

@app.route("/search")
def search():
    query = request.args.get("q", "")
    results = []
    if query:
        # 1. Search
        bm25_res = engine.search_bm25(query)
        # 2. Rerank
        reranked = engine.rerank(query, bm25_res)
        # 3. Aggregate
        results = engine.aggregate_results(reranked, query)
        
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
