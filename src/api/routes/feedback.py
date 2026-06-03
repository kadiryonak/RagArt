"""Feedback route — anonymous user feedback, appended to a local JSONL file.

Privacy: we collect ANONYMOUS feedback only (rating + free text + optional
question/answer context + a request id). No accounts, no PII required. The
UI shows a consent notice before sending. Storage is a single JSONL file
(data/feedback.jsonl) — zero external services, easy to inspect/aggregate.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from config.settings import settings
from src.api.schemas import FeedbackRequest, parse_body
from src.exceptions import ValidationError
from src.observability import get_request_id
from src.utils import StatusEmoji, get_logger

logger = get_logger(__name__)

bp = Blueprint("feedback", __name__)

# Serialise appends so concurrent requests don't interleave JSON lines.
_write_lock = threading.Lock()


def _feedback_path() -> Path:
    return Path(settings.DATA_FOLDER) / "feedback.jsonl"


@bp.route("/feedback", methods=["POST"])
def submit_feedback():
    """Store one anonymous feedback entry as a JSONL line."""
    body = parse_body(FeedbackRequest, request.get_json(silent=True))

    # At least one of rating / message must be present — empty feedback is noise.
    if body.rating is None and not (body.message or "").strip():
        raise ValidationError("Geri bildirim için bir puan veya mesaj gerekli.")

    entry = {
        "ts": time.time(),
        "request_id": get_request_id(),
        "rating": body.rating,
        "message": (body.message or "").strip(),
        "category": body.category,
        "question": body.question,
        "answer": body.answer,
        "page": body.page,
        "workspace": request.headers.get("X-Workspace-Id"),
        "ua": request.headers.get("User-Agent", "")[:300],
    }

    path = _feedback_path()
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # never fail the user over a feedback write
        logger.error(f"{StatusEmoji.ERROR} Feedback write failed: {e}")
        return jsonify({"error": "Geri bildirim kaydedilemedi."}), 500

    logger.info(f"{StatusEmoji.SUCCESS} Feedback saved (rating={body.rating})")
    return jsonify({"success": True, "message": "Teşekkürler! Geri bildiriminiz kaydedildi."})


@bp.route("/feedback/stats")
def feedback_stats():
    """Aggregate counts for the owner (avg rating, totals). Not user-facing."""
    path = _feedback_path()
    if not path.exists():
        return jsonify({"total": 0, "average_rating": None, "with_message": 0})

    total = 0
    rating_sum = 0
    rating_n = 0
    with_message = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                total += 1
                if isinstance(e.get("rating"), int):
                    rating_sum += e["rating"]
                    rating_n += 1
                if (e.get("message") or "").strip():
                    with_message += 1
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "total": total,
        "average_rating": round(rating_sum / rating_n, 2) if rating_n else None,
        "with_message": with_message,
    })
