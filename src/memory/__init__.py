"""Conversation memory plugins.

4 farklı strateji, hepsi BaseMemory sözleşmesini uygular:

    NoMemory               — varsayılan, hafıza yok
    SlidingWindowMemory    — son N turn (deterministik, zero-cost)
    SummaryBufferMemory    — eski turn'leri LLM ile özetle, son N'i tut
    VectorRetrievalMemory  — semantic retrieval over chat history

Tasarım kararı (stateless server):
    Server'da session tutmuyoruz. Client her /ask çağrısında full history
    yollayan, server o anlık stratejiyi uygulayan stateless model.
    BYOK pattern'iyle uyumlu — multi-tenant DB gerektirmez.
"""

from src.memory.base import BaseMemory, ConversationTurn
from src.memory.none import NoMemory
from src.memory.sliding_window import SlidingWindowMemory
from src.memory.summary_buffer import SummaryBufferMemory
from src.memory.vector_retrieval import VectorRetrievalMemory

__all__ = [
    "BaseMemory",
    "ConversationTurn",
    "NoMemory",
    "SlidingWindowMemory",
    "SummaryBufferMemory",
    "VectorRetrievalMemory",
]
