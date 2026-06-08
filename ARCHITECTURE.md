# Hybrid Fake News Detection Architecture

## Pipeline

1. User submits headline/body text in Flask UI
2. `app.py` sends the text to `HybridFakeNewsDetector`
3. The detector runs:
   - NLP preprocessing
   - TF-IDF vectorization
   - PassiveAggressive classification
   - confidence estimation
   - claim extraction
   - local RAG-style evidence retrieval
   - clickbait detection
   - source credibility scoring
   - live event verification for breaking earthquake claims
   - fact-check verification
   - weighted trust score calculation
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
  - claim extraction
  - local knowledge-base retrieval
  - clickbait detection
  - source credibility analysis
  - no-key live earthquake verification through USGS feeds
  - weighted trust score generation
  - explanation generation
  - external API fact-check hook
  - local knowledge base fallback

- `knowledge_base.json`
  Local fact-check knowledge base used when no external API is configured. At
  runtime this is indexed with a TF-IDF retriever so submitted claims can be
  matched against evidence semantically instead of using only exact keywords.

- `templates/index.html`
  Frontend dashboard.

- `model.pkl` and `vectorizer.pkl`
  Trained ML artifacts from notebook training.

## External Fact-Check API

Optional `.env` file at project root:

```env
FACT_CHECK_API_URL=https://your-api-endpoint
FACT_CHECK_API_KEY=your_api_key
LIVE_EVENT_CHECKS=true
USGS_EARTHQUAKE_FEED_URL=https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson
LIVE_NEWS_CHECKS=false
LIVE_NEWS_API_URL=https://api.gdeltproject.org/api/v2/doc/doc
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

## Advanced Hybrid Intelligence Layer

The current system is more than a single fake/real classifier. It combines four
signals into a final trust score:

| Signal | Weight | Purpose |
| --- | ---: | --- |
| ML model confidence | 35% | Measures whether the trained NLP model sees the text as real or fake. |
| Evidence retrieval | 35% | Retrieves similar verified claims from `knowledge_base.json` and checks whether they support or contradict the input. |
| Source credibility | 20% | Looks for URLs, HTTPS usage, reliable domains, suspicious domain wording, and publication metadata cues. |
| Clickbait risk | 10% | Penalizes sensational wording, excessive exclamation marks, and uppercase emphasis. |

For earthquake claims, the backend also checks the live USGS significant
earthquake feed before falling back to the local knowledge base. This helps the
system handle breaking events such as "Philippines hit a massive earthquake
today" even when the event has not yet been added to `knowledge_base.json`.

Additional API fields returned by `/api/analyze`:

```json
{
  "claims": [
    {
      "text": "India is the capital of Pakistan",
      "type": "factual_claim",
      "confidence": 80
    }
  ],
  "evidence": [
    {
      "claim": "India is the capital of Pakistan",
      "verdict": "False",
      "retrieval_score": 71.77
    }
  ],
  "source_credibility": {
    "score": 45,
    "risk_level": "Medium"
  },
  "trust_score": {
    "score": 35,
    "rating": "Very low trust"
  }
}
```

This lets the project handle factual contradictions such as "India is the
capital of Pakistan" even when the original writing-style classifier predicts
the text as real.

## Future BERT Upgrade

To switch from TF-IDF to BERT later:

1. Train and save a BERT classifier separately
2. Add a second model adapter class in `hybrid_detector.py`
3. Route inference through a `MODEL_BACKEND=tfidf|bert` configuration flag
4. Keep clickbait, fact-checking, explanation, and UI unchanged
