"""Memory base contract — stateless apply(history, query) → context string.

NEDEN STATELESS?
    Server hiç state tutmaz: BYOK pattern'iyle uyumlu, multi-tenant DB
    yok, session yönetimi yok. Client her istekte full history'i
    X-Conversation-History header'ında JSON olarak yollar.

NEDEN APPLY()?
    Tek metot, tek sorumluluk: history + current query alır, prompt'a
    eklenecek "memory context" string'ini döner. RAG context'inden
    AYRIDIR; LLM'e şu format gider:

        [MEMORY CONTEXT — varsa]
        [RAG CONTEXT — retrieved docs]
        SORU: {question}
        YANIT:
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal


Role = Literal["user", "assistant"]


@dataclass
class ConversationTurn:
    role: Role
    content: str

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        role = d.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        return cls(role=role, content=str(d.get("content", "")))

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


def format_turns(turns: List[ConversationTurn]) -> str:
    """Tek standart format: 'Kullanıcı:' / 'Asistan:'."""
    label = {"user": "Kullanıcı", "assistant": "Asistan"}
    return "\n".join(f"{label[t.role]}: {t.content}" for t in turns)


class BaseMemory(ABC):
    """Tüm hafıza stratejilerinin türetildiği temel sınıf."""

    name: str = "base"

    @abstractmethod
    def apply(self, history: List[ConversationTurn], query: str) -> str:
        """history + query → prompt'a eklenecek context string.

        Boş string döndürmesi "memory context yok" anlamına gelir; bu
        durumda LLM prompt'una memory bloğu eklenmez.
        """
