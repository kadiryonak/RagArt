"""Flask API'si için pydantic request şemaları.

NEDEN
    Route handler'ları `data.get("name", "").strip()` + elle
    `if not name: return 400` ile doluydu — validation 18 endpoint'e
    saçılmıştı, izole test edilemiyordu, hata mesajları tutarsızdı.

    Pydantic ile:
        - Şema tek yerde, tip-güvenli, ayrı ayrı test edilebilir
        - Geçersiz girdi tek bir RagArt ValidationError'a normalize olur
        - errors.py merkezi handler onu 400 + Türkçe mesaja map eder

KULLANIM
    body = parse_body(CreateWorkspaceRequest, request.get_json(silent=True))
    name = body.name          # garantili non-empty, strip'lenmiş

NOT
    parse_body() bilerek Flask'tan bağımsız: caller ham dict'i (genelde
    request.get_json(silent=True)) verir. Böylece şema mantığı Flask
    request context'i olmadan unit test edilebilir.
"""

from __future__ import annotations

from typing import Literal, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from src.exceptions import ValidationError

_T = TypeVar("_T", bound=BaseModel)


class _Schema(BaseModel):
    """Ortak taban: string alanları otomatik strip'le."""

    model_config = ConfigDict(str_strip_whitespace=True)


class AskRequest(_Schema):
    """POST /ask gövdesi.

    LLM ayarları (provider, model, params...) gövdede değil header'larda
    gelir — bkz. config.settings_schema.parse_request_settings.
    """

    question: str = Field(min_length=1, description="Kullanıcı sorusu")


class CreateWorkspaceRequest(_Schema):
    """POST /workspaces gövdesi."""

    name: str = Field(min_length=1, description="Çalışma alanı adı")
    color: Optional[str] = None
    description: str = ""
    vector_db: str = "chroma"


class UpdateWorkspaceRequest(_Schema):
    """PATCH /workspaces/<id> gövdesi — tüm alanlar opsiyonel."""

    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    vector_db: Optional[str] = None


class DeleteFileRequest(_Schema):
    """POST /delete-file gövdesi."""

    filename: str = Field(min_length=1, description="Silinecek dosya adı")


class CacheClearRequest(_Schema):
    """POST /cache/clear gövdesi.

    layer Literal ile kısıtlı; bilinmeyen katman pydantic seviyesinde
    reddedilir (route'ta ayrı bir kontrol gerekmez).
    """

    layer: Literal["all", "embedding", "response", "semantic"] = "all"


def parse_body(model_cls: Type[_T], data: object) -> _T:
    """Ham JSON gövdesini şemaya doğrula, tipli modeli döndür.

    Args:
        model_cls: doğrulanacak pydantic model sınıfı.
        data: ham gövde — genelde request.get_json(silent=True). None ise
            boş gövde sayılır (zorunlu alan yoksa geçer).

    Raises:
        ValidationError: gövde JSON nesnesi değilse ya da şema kontrolünden
            geçemezse. errors.py bunu 400'e map eder.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValidationError("İstek gövdesi bir JSON nesnesi olmalı.")
    try:
        return model_cls.model_validate(data)
    except PydanticValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(gövde)'}: {err['msg']}"
            for err in e.errors()
        )
        raise ValidationError(details) from e
