"""Workspaces — NotebookLM-style izole bilgi tabanları.

Her workspace:
    - Kendi dosya klasörü:      data/workspaces/{id}/files/
    - Kendi ChromaDB collection'ı (veya seçtiği DB)
    - Meta bilgileri:           data/workspaces/{id}/meta.json
                                (name, color, vector_db, created_at, file_count)

WorkspaceManager:
    - List, create, delete, switch işlemleri
    - İstek başına RAG instance'ını çözer (lazy-load)
    - Migration: eski data/*.json → workspaces/default/files/

Birden çok workspace açıkken her birinin embedding modeli paylaşılır
(memory tasarrufu); chroma_client'lar ayrı çünkü her workspace'in
kendi persist path'i var.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKSPACES_DIRNAME = "workspaces"
META_FILENAME = "meta.json"
FILES_DIRNAME = "files"
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_VECTOR_DB = "chroma"
COLOR_PALETTE = [
    "#6d5cf0", "#4f9a6a", "#c69143", "#c25a5a",
    "#4a90c2", "#9b6dc2", "#5fa896", "#c2965a",
]


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    """Workspace adından güvenli filesystem slug üret."""
    base = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not base:
        base = uuid.uuid4().hex[:8]
    return base[:48]


@dataclass
class Workspace:
    """Tek bir workspace'in metadata + lokasyon bilgisi."""
    id: str
    name: str
    color: str = COLOR_PALETTE[0]
    description: str = ""
    vector_db: str = DEFAULT_VECTOR_DB
    created_at: float = field(default_factory=time.time)
    file_count: int = 0
    last_modified: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Workspace":
        # Drop unknown fields rather than crashing (forward-compatibility)
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


class WorkspaceManager:
    """Workspaces'in tek-doğru-yer'i."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.workspaces_root = self.data_root / WORKSPACES_DIRNAME
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        # Migrate stray data/*.json into the default workspace on first run
        self._migrate_legacy_layout_if_needed()

    # ---------- Filesystem helpers ----------

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self.workspaces_root / workspace_id

    def files_dir(self, workspace_id: str) -> Path:
        d = self._workspace_dir(workspace_id) / FILES_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def chroma_path(self, workspace_id: str) -> Path:
        d = self._workspace_dir(workspace_id) / "chroma_db"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def vector_db_path(self, workspace_id: str, db_kind: str) -> Path:
        """Per-DB persist directory (chroma_db / qdrant_db / ...)."""
        sub = "chroma_db" if db_kind == "chroma" else f"{db_kind}_db"
        d = self._workspace_dir(workspace_id) / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def collection_name(self, workspace_id: str) -> str:
        # Keep collection names short + ASCII; the slug already is
        return f"ws_{workspace_id}_collection"

    def _meta_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / META_FILENAME

    # ---------- Migration ----------

    def _migrate_legacy_layout_if_needed(self) -> None:
        """data/*.json gibi eski dosyaları default workspace'e taşı."""
        if self._workspace_dir(DEFAULT_WORKSPACE_ID).exists():
            return  # Migration already done (or fresh install used default)

        # Discover legacy files in data/ root
        legacy_files: List[Path] = []
        if self.data_root.exists():
            for p in self.data_root.iterdir():
                if p.is_dir():
                    # Skip workspaces/ itself and known persistence dirs
                    continue
                if p.suffix.lower() in {".json", ".pdf", ".docx", ".md", ".markdown", ".txt"}:
                    legacy_files.append(p)

        # Create default workspace skeleton (even if no legacy files — we
        # want it present so the UI always has something to show)
        default_ws = Workspace(
            id=DEFAULT_WORKSPACE_ID,
            name="Default",
            description="Wikipedia örnek bilgi tabanı + ilk kullanım için.",
            vector_db=DEFAULT_VECTOR_DB,
        )
        self._write_meta(default_ws)
        files_target = self.files_dir(DEFAULT_WORKSPACE_ID)

        # Move legacy files (don't delete originals on different drives —
        # use copy + best-effort delete)
        for src in legacy_files:
            dest = files_target / src.name
            if dest.exists():
                continue
            try:
                shutil.move(str(src), str(dest))
            except Exception:
                try:
                    shutil.copy2(str(src), str(dest))
                except Exception:
                    continue

        # Update file count after migration
        default_ws.file_count = sum(1 for _ in files_target.iterdir() if _.is_file())
        self._write_meta(default_ws)

    # ---------- CRUD ----------

    def _read_meta(self, workspace_id: str) -> Optional[Workspace]:
        path = self._meta_path(workspace_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Workspace.from_dict(json.load(f))
        except Exception:
            return None

    def _write_meta(self, ws: Workspace) -> None:
        ws.last_modified = time.time()
        # Recompute file count cheaply
        files = self.files_dir(ws.id)
        ws.file_count = sum(1 for _ in files.iterdir() if _.is_file())
        path = self._meta_path(ws.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ws.to_dict(), f, ensure_ascii=False, indent=2)

    def list(self) -> List[Workspace]:
        out: List[Workspace] = []
        for child in sorted(self.workspaces_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            ws = self._read_meta(child.name)
            if ws is None:
                continue
            # Refresh file_count on every read so the UI is accurate
            files = self.files_dir(ws.id)
            ws.file_count = sum(1 for _ in files.iterdir() if _.is_file())
            out.append(ws)
        return out

    def get(self, workspace_id: str) -> Optional[Workspace]:
        return self._read_meta(workspace_id)

    def exists(self, workspace_id: str) -> bool:
        return self._meta_path(workspace_id).exists()

    def create(
        self,
        name: str,
        *,
        color: Optional[str] = None,
        description: str = "",
        vector_db: str = DEFAULT_VECTOR_DB,
    ) -> Workspace:
        if not name.strip():
            raise ValueError("Workspace name must not be empty.")

        slug = _slugify(name)
        # If duplicate, append a short uuid suffix
        candidate = slug
        i = 2
        while self.exists(candidate):
            candidate = f"{slug}-{i}"
            i += 1

        ws = Workspace(
            id=candidate,
            name=name.strip(),
            color=color or COLOR_PALETTE[hash(candidate) % len(COLOR_PALETTE)],
            description=description.strip(),
            vector_db=vector_db,
        )
        self._write_meta(ws)
        return ws

    def delete(self, workspace_id: str) -> bool:
        if workspace_id == DEFAULT_WORKSPACE_ID:
            # Never delete default — it's the migration target
            raise ValueError("Default workspace cannot be deleted.")
        target = self._workspace_dir(workspace_id)
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True

    def update(
        self,
        workspace_id: str,
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Workspace]:
        ws = self._read_meta(workspace_id)
        if ws is None:
            return None
        if name is not None:
            ws.name = name.strip() or ws.name
        if color is not None:
            ws.color = color
        if description is not None:
            ws.description = description.strip()
        self._write_meta(ws)
        return ws

    def touch(self, workspace_id: str) -> None:
        """Update last_modified after upload / reindex."""
        ws = self._read_meta(workspace_id)
        if ws is not None:
            self._write_meta(ws)

    def resolve(self, requested: Optional[str]) -> str:
        """Header'dan gelen workspace_id'yi validate et; yoksa default."""
        if requested and self.exists(requested):
            return requested
        # Ensure default exists (re-trigger migration if user deleted everything)
        if not self.exists(DEFAULT_WORKSPACE_ID):
            self._migrate_legacy_layout_if_needed()
        return DEFAULT_WORKSPACE_ID
