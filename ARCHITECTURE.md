# Hybrid Fake News Detection Architecture

## Pipeline

1. User submits headline/body text in Flask UI
2. `app.py` sends the text to `HybridFakeNewsDetector`
3. The detector runs:
   - NLP preprocessing
   - TF-IDF vectorization
   - PassiveAggressive classification
   - confidence estimation
   - clickbait detection
   - fact-check verification
   - explanation generation
4. Flask renders the full analysis dashboard

## Components

- `app.py`
  Flask entry point and route handling.

- `hybrid_detector.py`
  Core business logic for:
  - model loading
  - vectorizer loading
  - confidence scoring
  - clickbait detection
  - explanation generation
  - external API fact-check hook
  - local knowledge base fallback

- `knowledge_base.json`
  Local fact-check knowledge base used when no external API is configured.

- `templates/index.html`
  Frontend dashboard.

- `model.pkl` and `vectorizer.pkl`
  Trained ML artifacts from notebook training.

## External Fact-Check API

Optional `.env` file at project root:

```env
FACT_CHECK_API_URL=https://your-api-endpoint
FACT_CHECK_API_KEY=your_api_key
```

Expected API response format:

```json
{
  "claim": "matched claim",
  "verdict": "True",
  "explanation": "reason",
  "url": "https://source",
  "score": 92
}
```

## Future BERT Upgrade

To switch from TF-IDF to BERT later:

1. Train and save a BERT classifier separately
2. Add a second model adapter class in `hybrid_detector.py`
3. Route inference through a `MODEL_BACKEND=tfidf|bert` configuration flag
4. Keep clickbait, fact-checking, explanation, and UI unchanged
