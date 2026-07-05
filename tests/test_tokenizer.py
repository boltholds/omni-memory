from omni_memory.tokenizer import SimpleTokenizer, build_tokenizer

def test_simple_tokenizer_counts_words():
    tok = SimpleTokenizer()
    assert tok.count("a b  c") == 3
    assert tok.count("") == 0

def test_factory_auto_falls_back():
    tok = build_tokenizer(backend="auto", model_name="nonexistent")
    # должен вернуть какое-то число >= 0
    assert tok.count("hello world") >= 2
