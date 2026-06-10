"""Unit tests for src/pipeline/stages/groundedness.py."""

from __future__ import annotations

from src.pipeline.base import QueryRequest, QueryState
from src.pipeline.stages.groundedness import GroundednessStage


def _state(*, response=None, context="", answer=None):
    st = QueryState(request=QueryRequest(question="q"), rag=None)
    st.response = response
    st.context = context
    st.answer = answer
    return st


class TestGroundednessStage:
    def test_no_response_is_passthrough(self):
        st = _state(response=None)
        out = GroundednessStage()(st)
        assert out.response is None
        # __call__ still records timing
        assert "groundedness" in out.timings

    def test_non_rag_source_skipped(self):
        st = _state(response={"source": "cache"}, context="c", answer="a")
        out = GroundednessStage()(st)
        assert "groundedness_score" not in out.response

    def test_missing_context_skipped(self):
        st = _state(response={"source": "rag_system"}, context="", answer="cevap")
        out = GroundednessStage()(st)
        assert "groundedness_score" not in out.response

    def test_grounded_answer_scored_no_warning(self):
        text = "algoritma sıralı adımların listesidir"
        st = _state(response={"source": "rag_system"}, context=text, answer=text)
        out = GroundednessStage()(st)
        assert out.response["groundedness_score"] == 1.0
        assert "groundedness_warning" not in out.response

    def test_low_groundedness_adds_warning(self):
        st = _state(
            response={"source": "rag_system"},
            context="robotlar uzayda çalışır",
            answer="kediler bahçede koşar zıplar",
        )
        out = GroundednessStage()(st)
        assert out.response["groundedness_score"] < 0.3
        assert out.response["groundedness_warning"] is True

    def test_timing_always_recorded(self):
        st = _state(response={"source": "rag_system"}, context="x metni", answer="x metni")
        out = GroundednessStage()(st)
        assert isinstance(out.timings["groundedness"], float)

    def test_diacritic_folding_grounds_ocr_text(self):
        # Clean answer (with diacritics) vs OCR context (diacritics stripped).
        # Lexical overlap would collapse without folding; folding rescues it.
        st = _state(
            response={"source": "rag_system"},
            context="yapay ari kolonisi isci arilar gozcu arilar kasif arilar",
            answer="Yapay arı kolonisi: işçi arılar, gözcü arılar, kaşif arılar.",
        )
        out = GroundednessStage()(st)
        assert out.response["groundedness_score"] >= 0.3
        assert "groundedness_warning" not in out.response

    def test_semantic_rescues_low_lexical(self):
        # No shared tokens at all → lexical 0, but a stubbed semantic signal
        # (high cosine to the retrieved docs) lifts it over the threshold.
        class _Rag:
            def semantic_groundedness(self, answer, docs):
                return 0.71

        st = _state(
            response={"source": "rag_system"},
            context="bazı tamamen farklı sözcükler",
            answer="bambaşka kelimeler içeren cümle",
        )
        st.rag = _Rag()
        st.docs = [object()]  # non-empty → semantic path runs
        out = GroundednessStage()(st)
        assert out.response["groundedness_score"] == 0.71
        assert "groundedness_warning" not in out.response
