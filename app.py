import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from hybrid_detector import HybridFakeNewsDetector


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
detector = HybridFakeNewsDetector(BASE_DIR)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def build_analysis_payload(news_text: str):
    submitted_text = news_text.strip()
    if not submitted_text:
        return None, "Please enter a headline or article text."

    try:
        analysis = detector.analyze(submitted_text)
    except Exception as exc:
        return None, f"Analysis failed: {exc}"

    return {
        "input_text": submitted_text,
        "label": analysis["label"],
        "tone": analysis["tone"],
        "confidence": analysis["confidence"],
        "model_label": analysis["model_label"],
        "model_confidence": analysis["model_confidence"],
        "clickbait": analysis["clickbait"],
        "fact_check": analysis["fact_check"],
        "explanation": analysis["explanation"],
        "cleaned_text": analysis["cleaned_text"],
    }, None


@app.route("/", methods=["GET", "POST"])
def home():
    analysis = None
    submitted_text = ""
    error_message = None

    if request.method == "POST":
        submitted_text = request.form.get("news", "")
        analysis, error_message = build_analysis_payload(submitted_text)

    return render_template(
        "index.html",
        analysis=analysis,
        submitted_text=submitted_text,
        error_message=error_message,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_api():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    news_text = payload.get("text") or payload.get("news") or request.form.get("news", "")

    analysis, error_message = build_analysis_payload(news_text)
    if error_message:
        return jsonify({"ok": False, "error": error_message}), 400

    return jsonify({"ok": True, "analysis": analysis}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
