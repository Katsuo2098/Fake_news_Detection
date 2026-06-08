import os
from pathlib import Path

from flask import Flask, render_template, request

from hybrid_detector import HybridFakeNewsDetector


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
detector = HybridFakeNewsDetector(BASE_DIR)


@app.route("/", methods=["GET", "POST"])
def home():
    error_message = None
    analysis = None
    submitted_text = ""

    if request.method == "POST":
        submitted_text = request.form.get("news", "").strip()

        if not submitted_text:
            error_message = "Please enter a headline or article text."
        else:
            try:
                analysis = detector.analyze(submitted_text)
            except Exception as exc:
                error_message = f"Analysis failed: {exc}"

    return render_template(
        "index.html",
        analysis=analysis,
        submitted_text=submitted_text,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
