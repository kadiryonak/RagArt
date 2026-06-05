"""RelevanceFilterStage — drop irrelevant chunks before they reach the LLM.

Retrieval always returns top-k, but top-k can include chunks that merely share
a generic word with the question (a networking page that says "algoritma"
surfacing for an optimization question). Those distractors pollute the context,
hurt groundedness, and show up as misleading "sources". This stage removes them
per-document using a semantic + lexical bar (see
TurkishRAGSystem.filter_relevant_documents), so only genuinely-relevant chunks
continue to ContextStage / the gate / the sources panel.

Optional LLM judge: when request.relevance_judge is on AND a provider is
available, the surviving chunks are sent to the LLM ("which of these are
relevant to the question?") as a second, semantic pass — the user-requested
"ask the AI which chunks are related" fallback. Off by default (extra latency +
cost); enabled from Developer Mode.

Runs AFTER RetrievalStage and BEFORE RelevanceGateStage. It writes the filtered
list back to state.docs and stashes the best relevance score so the gate can
reuse it without recomputing embeddings.
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class RelevanceFilterStage(PipelineStage):
    name = "relevance_filter"

    def run(self, state: QueryState) -> QueryState:
        rag = state.rag
        if not state.docs:
            return state

        question = state.request.question
        kept, info = rag.filter_relevant_documents(question, state.docs)

        # Optional LLM judge — second, semantic pass over the survivors.
        if (
            getattr(state.request, "relevance_judge", False)
            and len(kept) > 1
        ):
            try:
                judged = rag.llm_judge_relevance(
                    question, kept,
                    llm_provider=state.request.llm_provider,
                    llm_params=dict(state.request.llm_params),
                )
                if judged:  # never let the judge empty the set silently
                    info["dropped"] += len(kept) - len(judged)
                    info["kept"] = len(judged)
                    info["judge"] = True
                    kept = judged
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("LLM relevance judge skipped: %s", e)

        if info.get("dropped"):
            logger.info(
                "Relevance filter: kept %d/%d chunk(s)%s",
                info["kept"], info["kept"] + info["dropped"],
                " (LLM judge)" if info.get("judge") else "",
            )

        state.docs = kept
        # Let the gate reuse the best score instead of recomputing embeddings.
        state.extra_meta["relevance_filter"] = info
        state.extra_meta["_relevance_max"] = info.get("max_score", 0.0)
        return state
