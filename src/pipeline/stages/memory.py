"""MemoryStage — sohbet hafızasını uygula.

Memory strategy adından (sliding_window / summary_buffer / vector / none)
bir Memory instance üretilir, history + sorgu üzerinde apply() çağrılır.
Sonuç string state.memory_context'e yazılır — ExecuteStage prompt'a
ekleyecek.

Default (none) → boş string → ExecuteStage memory bloğu olmadan davranır.
"""

from __future__ import annotations

from src.pipeline.base import PipelineStage, QueryState


class MemoryStage(PipelineStage):
    name = "memory"

    def run(self, state: QueryState) -> QueryState:
        rag = state.rag
        r = state.request
        memory = rag._build_memory(
            r.memory_strategy,
            llm_for_summary=r.llm_provider,
        )
        # request.history is a tuple; memory.apply expects a list
        state.memory_context = memory.apply(
            list(r.history), r.question,
        ).strip()
        return state
