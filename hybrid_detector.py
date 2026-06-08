from __future__ import annotations

import json
import math
import pickle
import re
import string
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridFakeNewsDetector:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.model = self._load_pickle(self.base_dir / "model.pkl")
        self.vectorizer = self._load_pickle(self.base_dir / "vectorizer.pkl")
        self.knowledge_base = self._load_knowledge_base(
            self.base_dir / "knowledge_base.json"
        )
        self.evidence_vectorizer, self.evidence_matrix = self._build_evidence_index()
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
        self.reliable_domains = {
            "apnews.com",
            "bbc.com",
            "bbc.co.uk",
            "britannica.com",
            "cdc.gov",
            "climate.nasa.gov",
            "nasa.gov",
            "reuters.com",
            "who.int",
        }
        self.unreliable_domain_terms = {
            "click",
            "dailybuzz",
            "exposed",
            "insidertruth",
            "rumor",
            "viral",
        }

    def analyze(self, text: str) -> dict[str, Any]:
        cleaned_text = self.clean_text(text)
        claims = self._extract_claims(text)
        vector = self.vectorizer.transform([cleaned_text])
        prediction_value = self.model.predict(vector)[0]
        model_label = self._normalize_label(prediction_value)
        confidence_score = self._estimate_confidence(vector, prediction_value)
        clickbait = self._detect_clickbait(text)
        source_credibility = self._score_source_credibility(text)
        evidence_matches = self._retrieve_evidence(claims or [text])
        fact_check = self._fact_check(text, cleaned_text, evidence_matches)
        trust_score = self._calculate_trust_score(
            model_label=model_label,
            confidence_score=confidence_score,
            fact_check=fact_check,
            clickbait=clickbait,
            source_credibility=source_credibility,
        )
        explanation = self._build_explanation(
            text=text,
            cleaned_text=cleaned_text,
            claims=claims,
            vector=vector,
            model_label=model_label,
            confidence_score=confidence_score,
            clickbait=clickbait,
            fact_check=fact_check,
            source_credibility=source_credibility,
            trust_score=trust_score,
        )

        hybrid_label = model_label
        hybrid_confidence = confidence_score
        status_tone = "real"

        if fact_check["status"] == "contradicted":
            hybrid_label = "Fake News"
            hybrid_confidence = max(hybrid_confidence, 0.88)
            status_tone = "fake"
        elif trust_score["score"] < 45:
            hybrid_label = "Fake News"
            hybrid_confidence = max(hybrid_confidence, (100 - trust_score["score"]) / 100)
            status_tone = "fake"
        elif fact_check["status"] == "supported" and model_label == "Real News":
            hybrid_confidence = max(hybrid_confidence, 0.80)
            status_tone = "real"
        elif model_label == "Fake News":
            status_tone = "fake"

        if (
            clickbait["score"] >= 0.45
            or source_credibility["risk_level"] == "High"
        ) and hybrid_label == "Real News":
            status_tone = "warning"

        return {
            "label": hybrid_label,
            "tone": status_tone,
            "confidence": round(hybrid_confidence * 100, 2),
            "model_label": model_label,
            "model_confidence": round(confidence_score * 100, 2),
            "clickbait": clickbait,
            "fact_check": fact_check,
            "claims": claims,
            "evidence": evidence_matches,
            "source_credibility": source_credibility,
            "trust_score": trust_score,
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

    def _build_evidence_index(self):
        if not self.knowledge_base:
            return None, None

        documents = []
        for entry in self.knowledge_base:
            documents.append(
                " ".join(
                    [
                        str(entry.get("claim", "")),
                        str(entry.get("verdict", "")),
                        str(entry.get("explanation", "")),
                    ]
                )
            )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(documents)
        return vectorizer, matrix

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

    def _extract_claims(self, text: str) -> list[dict[str, Any]]:
        text_without_urls = re.sub(r"https?://\S+|www\.\S+", " ", text)
        normalized = re.sub(r"\s+", " ", text_without_urls.strip())
        if not normalized:
            return []

        sentence_candidates = re.split(r"(?<=[.!?])\s+|[\n;]+", normalized)
        claims = []

        claim_keywords = {
            "announced",
            "banned",
            "caused",
            "confirmed",
            "contains",
            "cures",
            "discovered",
            "is",
            "killed",
            "proves",
            "said",
            "shows",
            "was",
            "will",
        }

        for sentence in sentence_candidates:
            candidate = sentence.strip(" -")
            words = candidate.split()
            if len(words) < 4:
                continue

            lowered = candidate.lower()
            has_number = bool(re.search(r"\b\d{2,4}\b", candidate))
            has_named_entity_hint = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", candidate))
            has_claim_verb = any(
                re.search(rf"\b{re.escape(keyword)}\b", lowered)
                for keyword in claim_keywords
            )

            if has_number or has_named_entity_hint or has_claim_verb:
                claims.append(
                    {
                        "text": candidate,
                        "type": "factual_claim",
                        "confidence": self._claim_confidence(
                            has_number=has_number,
                            has_named_entity_hint=has_named_entity_hint,
                            has_claim_verb=has_claim_verb,
                            word_count=len(words),
                        ),
                    }
                )

        if not claims and len(normalized.split()) >= 4:
            claims.append(
                {
                    "text": normalized,
                    "type": "possible_claim",
                    "confidence": 45.0,
                }
            )

        claims.sort(key=lambda item: item["confidence"], reverse=True)
        return claims[:4]

    def _claim_confidence(
        self,
        has_number: bool,
        has_named_entity_hint: bool,
        has_claim_verb: bool,
        word_count: int,
    ) -> float:
        score = 35.0
        if has_number:
            score += 20.0
        if has_named_entity_hint:
            score += 20.0
        if has_claim_verb:
            score += 20.0
        if 6 <= word_count <= 35:
            score += 5.0
        return round(min(score, 95.0), 2)

    def _retrieve_evidence(self, claims: list[dict[str, Any]] | list[str]) -> list[dict[str, Any]]:
        if self.evidence_vectorizer is None or self.evidence_matrix is None:
            return []

        queries = [
            claim["text"] if isinstance(claim, dict) else claim
            for claim in claims
            if (claim["text"] if isinstance(claim, dict) else claim).strip()
        ]
        if not queries:
            return []

        retrieved: dict[int, dict[str, Any]] = {}

        for query in queries:
            query_vector = self.evidence_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.evidence_matrix)[0]
            ranked_indices = similarities.argsort()[::-1][:3]

            for index in ranked_indices:
                score = float(similarities[index])
                if score < 0.10:
                    continue

                entry = self.knowledge_base[int(index)]
                current = retrieved.get(int(index))
                match = {
                    "claim": entry.get("claim", ""),
                    "verdict": entry.get("verdict", "Unknown"),
                    "explanation": entry.get("explanation", ""),
                    "source_url": entry.get("source_url", ""),
                    "retrieval_score": round(score * 100, 2),
                    "matched_query": query,
                }
                if current is None or match["retrieval_score"] > current["retrieval_score"]:
                    retrieved[int(index)] = match

        matches = sorted(
            retrieved.values(),
            key=lambda item: item["retrieval_score"],
            reverse=True,
        )
        return matches[:5]

    def _fact_check(
        self,
        raw_text: str,
        cleaned_text: str,
        evidence_matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        api_result = self._fact_check_api(raw_text)
        if api_result is not None:
            return api_result

        best_match = evidence_matches[0] if evidence_matches else None
        best_score = best_match["retrieval_score"] if best_match else 0.0

        if best_match and best_score >= 35:
            verdict = best_match["verdict"].lower()
            if verdict in {"true", "real", "supported"}:
                status = "supported"
            elif verdict in {"false", "fake", "debunked"}:
                status = "contradicted"
            else:
                status = "unverified"

            return {
                "source": "Local RAG knowledge base",
                "status": status,
                "matched_claim": best_match["claim"],
                "verdict": best_match["verdict"],
                "explanation": best_match["explanation"],
                "evidence_url": best_match.get("source_url", ""),
                "match_score": round(best_score, 2),
            }

        lexical_match = self._lexical_fact_check(cleaned_text)
        if lexical_match is not None:
            return lexical_match

        return {
            "source": "Local RAG knowledge base",
            "status": "unverified",
            "matched_claim": None,
            "verdict": "No close fact-check match",
            "explanation": (
                "No close semantic match was found in the local fact-check knowledge base. "
                "You can connect an external fact-check API with environment variables "
                "for broader verification."
            ),
            "evidence_url": "",
            "match_score": 0.0,
        }

    def _lexical_fact_check(self, cleaned_text: str) -> dict[str, Any] | None:
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

        if not best_match or best_score < 0.45:
            return None

        verdict = best_match["verdict"].lower()
        if verdict in {"true", "real", "supported"}:
            status = "supported"
        elif verdict in {"false", "fake", "debunked"}:
            status = "contradicted"
        else:
            status = "unverified"

        return {
            "source": "Local lexical knowledge base",
            "status": status,
            "matched_claim": best_match["claim"],
            "verdict": best_match["verdict"],
            "explanation": best_match["explanation"],
            "evidence_url": best_match.get("source_url", ""),
            "match_score": round(best_score * 100, 2),
        }

    def _score_source_credibility(self, text: str) -> dict[str, Any]:
        urls = re.findall(r"https?://[^\s)>\]]+", text)
        domains = []
        reasons = []
        score = 55.0

        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().removeprefix("www.")
            if not domain:
                continue

            domains.append(domain)
            if parsed.scheme == "https":
                score += 5.0
            else:
                score -= 10.0
                reasons.append(f"{domain} does not use HTTPS.")

            if self._domain_in_set(domain, self.reliable_domains):
                score += 25.0
                reasons.append(f"{domain} appears in the reliable-source list.")

            if any(term in domain for term in self.unreliable_domain_terms):
                score -= 25.0
                reasons.append(f"{domain} has suspicious wording in the domain name.")

        if not urls:
            reasons.append("No source URL was provided, so source reputation could not be verified.")
            score -= 10.0

        if re.search(r"\b(author|byline|published|updated)\b", text.lower()):
            score += 5.0
            reasons.append("The text contains publication metadata cues.")

        score = max(0.0, min(score, 100.0))
        if score >= 75:
            risk_level = "Low"
        elif score >= 45:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "score": round(score, 2),
            "risk_level": risk_level,
            "domains": sorted(set(domains)),
            "signals": reasons,
        }

    def _domain_in_set(self, domain: str, candidates: set[str]) -> bool:
        return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)

    def _calculate_trust_score(
        self,
        model_label: str,
        confidence_score: float,
        fact_check: dict[str, Any],
        clickbait: dict[str, Any],
        source_credibility: dict[str, Any],
    ) -> dict[str, Any]:
        model_component = (
            confidence_score * 100 if model_label == "Real News" else (1 - confidence_score) * 100
        )

        if fact_check["status"] == "supported":
            evidence_component = 85 + min(fact_check["match_score"], 100) * 0.15
        elif fact_check["status"] == "contradicted":
            evidence_component = 100 - min(fact_check["match_score"], 100)
        else:
            evidence_component = 50

        clickbait_component = 100 - clickbait["score_percent"]
        source_component = source_credibility["score"]

        score = (
            model_component * 0.35
            + evidence_component * 0.35
            + source_component * 0.20
            + clickbait_component * 0.10
        )

        if fact_check["status"] == "contradicted":
            score = min(score, 35.0)
        elif fact_check["status"] == "supported":
            score = max(score, 65.0)

        if score >= 75:
            rating = "High trust"
        elif score >= 55:
            rating = "Moderate trust"
        elif score >= 40:
            rating = "Low trust"
        else:
            rating = "Very low trust"

        return {
            "score": round(max(0.0, min(score, 100.0)), 2),
            "rating": rating,
            "components": {
                "model": round(model_component, 2),
                "evidence": round(evidence_component, 2),
                "source": round(source_component, 2),
                "clickbait": round(clickbait_component, 2),
            },
            "weights": {
                "model": 0.35,
                "evidence": 0.35,
                "source": 0.20,
                "clickbait": 0.10,
            },
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
        claims: list[dict[str, Any]],
        vector,
        model_label: str,
        confidence_score: float,
        clickbait: dict[str, Any],
        fact_check: dict[str, Any],
        source_credibility: dict[str, Any],
        trust_score: dict[str, Any],
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
                "The retrieval-based verification layer found supporting evidence."
            )
        elif fact_check["status"] == "contradicted":
            reasons.append(
                "The retrieval-based verification layer found contradicting evidence."
            )
        else:
            reasons.append("No strong fact-check match was found for this claim.")

        if claims:
            reasons.append(
                f"Claim extraction found {len(claims)} checkable claim"
                f"{'' if len(claims) == 1 else 's'}."
            )

        reasons.append(
            f"Source credibility was rated {source_credibility['risk_level'].lower()} "
            f"with a {source_credibility['score']}% score."
        )
        reasons.append(
            f"The final weighted trust score is {trust_score['score']}% "
            f"({trust_score['rating'].lower()})."
        )

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
