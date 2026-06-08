from __future__ import annotations

import json
import math
import pickle
import re
import string
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HybridFakeNewsDetector:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.model = self._load_pickle(self.base_dir / "model.pkl")
        self.vectorizer = self._load_pickle(self.base_dir / "vectorizer.pkl")
        self.knowledge_base = self._load_knowledge_base(
            self.base_dir / "knowledge_base.json"
        )
        self.clickbait_terms = {
            "shocking",
            "breaking",
            "you won't believe",
            "jaw-dropping",
            "exclusive",
            "must see",
            "unbelievable",
            "viral",
            "secret",
            "miracle",
            "bombshell",
            "stunning",
            "what happened next",
            "the truth about",
            "exposed",
            "urgent",
            "alert",
            "amazing",
            "sensational",
            "controversial",
        }

    def analyze(self, text: str) -> dict[str, Any]:
        cleaned_text = self.clean_text(text)
        vector = self.vectorizer.transform([cleaned_text])
        prediction_value = self.model.predict(vector)[0]
        model_label = self._normalize_label(prediction_value)
        confidence_score = self._estimate_confidence(vector, prediction_value)
        clickbait = self._detect_clickbait(text)
        fact_check = self._fact_check(text, cleaned_text)
        explanation = self._build_explanation(
            text=text,
            cleaned_text=cleaned_text,
            vector=vector,
            model_label=model_label,
            confidence_score=confidence_score,
            clickbait=clickbait,
            fact_check=fact_check,
        )

        hybrid_label = model_label
        hybrid_confidence = confidence_score
        status_tone = "real"

        if fact_check["status"] == "contradicted":
            hybrid_label = "Fake News"
            hybrid_confidence = max(hybrid_confidence, 0.88)
            status_tone = "fake"
        elif fact_check["status"] == "supported" and model_label == "Real News":
            hybrid_confidence = max(hybrid_confidence, 0.80)
            status_tone = "real"
        elif model_label == "Fake News":
            status_tone = "fake"

        if clickbait["score"] >= 0.45 and hybrid_label == "Real News":
            status_tone = "warning"

        return {
            "label": hybrid_label,
            "tone": status_tone,
            "confidence": round(hybrid_confidence * 100, 2),
            "model_label": model_label,
            "model_confidence": round(confidence_score * 100, 2),
            "clickbait": clickbait,
            "fact_check": fact_check,
            "explanation": explanation,
            "cleaned_text": cleaned_text,
        }

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<.*?>+", " ", text)
        text = re.sub(r"\w*\d\w*", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _load_pickle(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path.name} not found at {file_path}. "
                "Place it in the project root folder."
            )

        with file_path.open("rb") as file:
            return pickle.load(file)

    def _load_knowledge_base(self, file_path: Path) -> list[dict[str, Any]]:
        if not file_path.exists():
            return []

        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _normalize_label(self, prediction_value: Any) -> str:
        if str(prediction_value).upper() == "FAKE" or prediction_value == 0:
            return "Fake News"
        return "Real News"

    def _estimate_confidence(self, vector, prediction_value: Any) -> float:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            return float(max(probabilities))

        if hasattr(self.model, "decision_function"):
            raw_score = self.model.decision_function(vector)
            if hasattr(raw_score, "__len__"):
                raw_score = raw_score[0]
            probability_real = 1 / (1 + math.exp(-float(raw_score)))
            if str(prediction_value).upper() == "FAKE" or prediction_value == 0:
                return max(1 - probability_real, probability_real)
            return max(probability_real, 1 - probability_real)

        return 0.5

    def _detect_clickbait(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        matched_terms = sorted(
            term for term in self.clickbait_terms if term in lowered
        )
        exclamation_count = text.count("!")
        uppercase_words = re.findall(r"\b[A-Z]{3,}\b", text)

        score = min(
            1.0,
            len(matched_terms) * 0.18
            + min(exclamation_count, 4) * 0.08
            + min(len(uppercase_words), 4) * 0.08,
        )

        if score >= 0.55:
            risk_level = "High"
        elif score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "score": round(score, 2),
            "score_percent": round(score * 100, 2),
            "risk_level": risk_level,
            "matched_terms": matched_terms,
            "signals": {
                "exclamation_count": exclamation_count,
                "uppercase_words": uppercase_words[:6],
            },
        }

    def _fact_check(self, raw_text: str, cleaned_text: str) -> dict[str, Any]:
        api_result = self._fact_check_api(raw_text)
        if api_result is not None:
            return api_result

        best_match = None
        best_score = 0.0
        input_tokens = set(cleaned_text.split())

        for entry in self.knowledge_base:
            claim_tokens = set(self.clean_text(entry["claim"]).split())
            if not claim_tokens:
                continue
            overlap = len(input_tokens & claim_tokens) / len(claim_tokens)
            if overlap > best_score:
                best_score = overlap
                best_match = entry

        if best_match and best_score >= 0.45:
            verdict = best_match["verdict"].lower()
            if verdict in {"true", "real", "supported"}:
                status = "supported"
            elif verdict in {"false", "fake", "debunked"}:
                status = "contradicted"
            else:
                status = "unverified"

            return {
                "source": "Local knowledge base",
                "status": status,
                "matched_claim": best_match["claim"],
                "verdict": best_match["verdict"],
                "explanation": best_match["explanation"],
                "evidence_url": best_match.get("source_url", ""),
                "match_score": round(best_score * 100, 2),
            }

        return {
            "source": "Local knowledge base",
            "status": "unverified",
            "matched_claim": None,
            "verdict": "No close fact-check match",
            "explanation": (
                "No close match was found in the local fact-check knowledge base. "
                "You can connect an external fact-check API with environment variables "
                "for broader verification."
            ),
            "evidence_url": "",
            "match_score": 0.0,
        }

    def _fact_check_api(self, raw_text: str) -> dict[str, Any] | None:
        api_url = self._read_env_file().get("FACT_CHECK_API_URL", "")
        api_key = self._read_env_file().get("FACT_CHECK_API_KEY", "")

        if not api_url:
            return None

        payload = json.dumps({"text": raw_text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = Request(api_url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return {
                "source": "External fact-check API",
                "status": "unverified",
                "matched_claim": None,
                "verdict": "API unavailable",
                "explanation": (
                    "The configured external fact-check API could not be reached, "
                    "so the system fell back to the local knowledge base."
                ),
                "evidence_url": "",
                "match_score": 0.0,
            }

        verdict = str(body.get("verdict", "unverified")).lower()
        if verdict in {"true", "real", "supported"}:
            status = "supported"
        elif verdict in {"false", "fake", "debunked"}:
            status = "contradicted"
        else:
            status = "unverified"

        return {
            "source": "External fact-check API",
            "status": status,
            "matched_claim": body.get("claim"),
            "verdict": body.get("verdict", "Unknown"),
            "explanation": body.get("explanation", "No explanation returned."),
            "evidence_url": body.get("url", ""),
            "match_score": float(body.get("score", 0)),
        }

    def _read_env_file(self) -> dict[str, str]:
        env_path = self.base_dir / ".env"
        values: dict[str, str] = {}

        if not env_path.exists():
            return values

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    def _build_explanation(
        self,
        text: str,
        cleaned_text: str,
        vector,
        model_label: str,
        confidence_score: float,
        clickbait: dict[str, Any],
        fact_check: dict[str, Any],
    ) -> dict[str, Any]:
        top_terms = self._top_weighted_terms(vector)
        reasons = []

        reasons.append(
            f"The NLP classifier predicted {model_label.lower()} with "
            f"{round(confidence_score * 100, 2)}% confidence."
        )

        if top_terms:
            reasons.append(
                "Most influential text features: " + ", ".join(top_terms) + "."
            )

        if clickbait["matched_terms"]:
            reasons.append(
                "Clickbait indicators detected: "
                + ", ".join(clickbait["matched_terms"])
                + "."
            )
        elif clickbait["risk_level"] == "Low":
            reasons.append("The text uses limited clickbait-style wording.")

        if fact_check["status"] == "supported":
            reasons.append(
                "Fact-checking found a supporting claim in the verification layer."
            )
        elif fact_check["status"] == "contradicted":
            reasons.append(
                "Fact-checking found a contradiction in the verification layer."
            )
        else:
            reasons.append("No strong fact-check match was found for this claim.")

        short_summary = " ".join(reasons[:3])

        return {
            "summary": short_summary,
            "reasons": reasons,
            "top_terms": top_terms,
            "text_length": len(text.split()),
            "cleaned_preview": " ".join(cleaned_text.split()[:40]),
        }

    def _top_weighted_terms(self, vector) -> list[str]:
        if not hasattr(self.model, "coef_"):
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = vector.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        weights = []
        coefficients = self.model.coef_[0]
        for idx in nonzero_indices:
            contribution = float(vector[0, idx]) * abs(float(coefficients[idx]))
            weights.append((contribution, feature_names[idx]))

        weights.sort(reverse=True)
        return [term for _, term in weights[:6]]
