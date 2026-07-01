"""Provenance Guard — Flask backend.

Endpoints:
  POST /submit   -> classify content, return attribution + confidence + label
  POST /appeal   -> contest a classification, flip status to under_review
  GET  /log      -> recent structured audit entries
  GET  /health   -> liveness probe

See planning.md for the full architecture and design rationale.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import db
import scoring
from signals import llm_signal, structural_signal, lexical_signal

app = Flask(__name__)

# Rate limiting — see README "Rate limiting" for the reasoning behind these numbers.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

db.init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@app.errorhandler(429)
def ratelimit_handler(e):
    return (
        jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": f"Too many submissions: {e.description}. Please slow down.",
            }
        ),
        429,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "provenance-guard"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    creator_id = (body.get("creator_id") or "").strip()

    if not text:
        return jsonify({"error": "bad_request", "message": "`text` is required"}), 400
    if not creator_id:
        return jsonify({"error": "bad_request", "message": "`creator_id` is required"}), 400

    # --- detection pipeline -------------------------------------------------
    llm = llm_signal(text)
    structural = structural_signal(text)
    lexical = lexical_signal(text)
    result = scoring.combine(llm, structural, lexical)
    label = scoring.make_label(result["confidence"])

    content_id = str(uuid.uuid4())
    timestamp = _now()
    status = "classified"

    signals_payload = {
        "llm": {
            "ai_likelihood": llm["ai_likelihood"],
            "reliable": llm["reliable"],
            "detail": llm["detail"],
        },
        "structural": {
            "ai_likelihood": structural["ai_likelihood"],
            "reliable": structural["reliable"],
            "detail": structural["detail"],
        },
        "lexical": {
            "ai_likelihood": lexical["ai_likelihood"],
            "reliable": lexical["reliable"],
            "detail": lexical["detail"],
        },
        "weights": result["weights"],
        "disagreement": result["disagreement"],
    }

    # --- persist ------------------------------------------------------------
    db.save_content(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "llm_score": llm["ai_likelihood"],
            "structural_score": structural["ai_likelihood"],
            "lexical_score": lexical["ai_likelihood"],
            "status": status,
            "created_at": timestamp,
        }
    )
    db.add_audit_entry(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": timestamp,
            "event_type": "classified",
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "llm_score": llm["ai_likelihood"],
            "structural_score": structural["ai_likelihood"],
            "lexical_score": lexical["ai_likelihood"],
            "status": status,
            "detail": signals_payload,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "signals": signals_payload,
            "label": label,
            "status": status,
            "timestamp": timestamp,
        }
    )


@app.post("/appeal")
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = (body.get("content_id") or "").strip()
    reasoning = (body.get("creator_reasoning") or "").strip()

    if not content_id:
        return jsonify({"error": "bad_request", "message": "`content_id` is required"}), 400
    if not reasoning:
        return (
            jsonify({"error": "bad_request", "message": "`creator_reasoning` is required"}),
            400,
        )

    content = db.get_content(content_id)
    if content is None:
        return (
            jsonify({"error": "not_found", "message": f"No content with id {content_id}"}),
            404,
        )

    new_status = "under_review"
    timestamp = _now()
    db.update_status(content_id, new_status)

    # Log the appeal alongside the original decision's scores.
    db.add_audit_entry(
        {
            "content_id": content_id,
            "creator_id": content["creator_id"],
            "timestamp": timestamp,
            "event_type": "appeal",
            "attribution": content["attribution"],
            "confidence": content["confidence"],
            "llm_score": content["llm_score"],
            "structural_score": content["structural_score"],
            "lexical_score": content["lexical_score"],
            "status": new_status,
            "appeal_reasoning": reasoning,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "status": new_status,
            "message": (
                "Appeal received. This content is now under review by a human moderator. "
                "The original classification has been recorded alongside your reasoning."
            ),
            "original_attribution": content["attribution"],
            "original_confidence": content["confidence"],
            "timestamp": timestamp,
        }
    )


@app.get("/log")
def log():
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 500))
    return jsonify({"entries": db.get_log(limit)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
