from app.entities import RegexEntityExtractor, EntityLinker

def test_regex_extractor_basic():
    extr = RegexEntityExtractor()
    out = extr.extract("Alice met Nikolai near the lighthouse.")
    assert "alice" in out and "nikolai" in out and "lighthouse" in out

def test_alias_linking():
    linker = EntityLinker({"lighthouse":["beacon","phare"], "alice":["alisa","алиса"]})
    assert linker.link_one("BEACON") == "lighthouse"
    assert linker.link_one("Алиса") == "alice"
    seq = linker.link_all(["BEACON","phare","lighthouse"])
    assert seq == ["lighthouse"]  # дедуп и канонизация
