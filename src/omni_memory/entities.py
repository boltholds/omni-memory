# app/entities.py
from __future__ import annotations
import re
import unicodedata
from typing import Iterable, List, Dict, Set, Protocol

TokenRe = re.compile(r"[A-Za-zА-Яа-яЁё0-9][\w\-']{1,}", re.UNICODE)

def _norm(s: str) -> str:
    # простая канонизация: NFKC -> lower -> strip punct
    t = unicodedata.normalize("NFKC", s).lower().strip()
    # уберём хвостовую пунктуацию
    return t.strip(".,:;!?\"'()[]{}")

class EntityExtractor(Protocol):
    def extract(self, text: str) -> List[str]: ...

class RegexEntityExtractor:
    """
    Простой извлекатель: слова длиной >= 3, цифро-буквенные, нормализуем.
    """
    def __init__(self, min_len: int = 3) -> None:
        self.min_len = min_len

    def extract(self, text: str) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for m in TokenRe.finditer(text):
            tok = _norm(m.group(0))
            if len(tok) < self.min_len:
                continue
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

class SpacyEntityExtractor:
    """
    Опционально: spaCy NER. Если нет — кинет RuntimeError.
    Извлекаем PERSON/ORG/GPE/LOC/PRODUCT/WORK_OF_ART и fallback на Regex.
    """
    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy  # type: ignore
        except Exception as e:
            raise RuntimeError("spaCy недоступен. Установи spacy и модель.") from e
        try:
            self.nlp = spacy.load(model)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель spaCy: {model}") from e
        self.regex = RegexEntityExtractor()

    def extract(self, text: str) -> List[str]:
        doc = self.nlp(text)
        ents = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART"}:
                ents.append(_norm(ent.text))
        # + добавим regex для коротких/пропущенных токенов
        ents += self.regex.extract(text)
        # uniq с сохранением порядка
        seen: Set[str] = set()
        out: List[str] = []
        for e in ents:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

class EntityLinker:
    """
    Алиасы: маппинг canonical -> [aliases...].
    build_index() строит обратный индекс alias -> canonical.
    link_all() нормализует список сущностей к канонам + добавляет канон, если пришёл алиас.
    """
    def __init__(self, aliases: Dict[str, List[str]]) -> None:
        self.aliases = { _norm(k): [ _norm(v) for v in vs ] for k,vs in aliases.items() }
        self.rev: Dict[str, str] = {}
        for canon, vs in self.aliases.items():
            self.rev[canon] = canon
            for a in vs:
                self.rev[a] = canon

    def link_one(self, e: str) -> str:
        e = _norm(e)
        return self.rev.get(e, e)

    def link_all(self, ents: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for e in ents:
            canon = self.link_one(e)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out

def build_entity_stack(backend: str, aliases: Dict[str, List[str]]):
    if backend == "spacy":
        try:
            extractor = SpacyEntityExtractor()
        except Exception:
            extractor = RegexEntityExtractor()
    elif backend == "auto":
        try:
            extractor = SpacyEntityExtractor()
        except Exception:
            extractor = RegexEntityExtractor()
    else:
        extractor = RegexEntityExtractor()
    linker = EntityLinker(aliases or {})
    return extractor, linker
