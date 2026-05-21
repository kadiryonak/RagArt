"""ContextStage — retrieved doc'ları LLM prompt'una uygun stringe çevir.

Format:
    [Source 1 - algoritma.json]
    {content}

    [Source 2 - python.json]
    {content}

Her source bloğu çift newline ile ayrılır. Source etiketleri prompt
içinde LLM'e görünür ama default DirectStrategy bunları cevapta
yazmamasını söylüyor.

NOT: Bu stage state.docs YOK ise hiçbir şey yapmaz — RetrievalStage'in
docs üretmesi gerekir; üretmediyse zaten önceki stage (RelevanceGate veya
benzeri) state.response set etmiş olur.
"""

from __future__ import annotations

from src.pipeline.base import PipelineStage, QueryState


class ContextStage(PipelineStage):
    name = "context"

    def run(self, state: QueryState) -> QueryState:
        if not state.docs:
            state.context = ""
            return state

        parts = []
        for i, doc in enumerate(state.docs, 1):
            source = doc.metadata.get("source", "Unknown")
            parts.append(f"[Source {i} - {source}]\n{doc.page_content}")
        state.context = "\n\n".join(parts)
        return state
