class VectorRepoError(Exception):
    """База для ошибок VectorStoreRepo."""
    ...

class CapacityExceeded(VectorRepoError):
    """Достигнут предел вместимости репозитория."""
    ...

class EmbedderDimMismatch(VectorRepoError):
    """Размерность эмбеддера не совпала со снапшотом."""
    ...

class SnapshotCorrupted(VectorRepoError):
    """Снапшот существует, но повреждён/нечитаем."""
    ...

class PersistenceError(VectorRepoError):
    """Ошибка сохранения/загрузки индекса/метаданных."""
    ...


class EpisodicRepoError(Exception):
    """База для ошибок EpisodicRepo."""
    ...

class SchemaInitError(EpisodicRepoError):
    """Не удалось инициализировать схему БД."""
    ...

class PersistenceError(EpisodicRepoError):
    """Ошибка обращения к БД (вставка/поиск/удаление)."""
    ...

class DataIntegrityError(EpisodicRepoError):
    """Повреждённые данные в БД (невалидный JSON и т.п.)."""
    ...

