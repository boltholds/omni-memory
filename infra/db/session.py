from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from infra.db.base import Base
from infra.db import models  # noqa: F401 - ensure metadata is populated


class DatabaseHandle:
    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self.engine = create_engine(database_url, echo=echo, future=True)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
