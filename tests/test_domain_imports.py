def test_imports():
    from domain import models, ports

    _ = models.MemoryObject
    _ = models.Fact
    _ = models.Episode
    _ = models.RetrievalBundle
    _ = models.ContextPack
    _ = ports.IMemoryOrchestrator
    _ = ports.IRetriever
