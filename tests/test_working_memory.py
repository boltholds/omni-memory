from omni_memory.domain.working_memory import WorkingMemory

def test_append_and_truncate():
    wm = WorkingMemory(max_tokens=5)
    wm.append("one two three four five")
    wm.append("six seven")
    # Должно остаться только последние 5 слов
    result = wm.to_prompt().split()
    assert result == ["three", "four", "five", "six", "seven"]

def test_no_truncate_if_under_limit():
    wm = WorkingMemory(max_tokens=10)
    wm.append("hello world")
    assert "hello world" in wm.to_prompt()

def test_multiple_appends_keep_tail():
    wm = WorkingMemory(max_tokens=4)
    wm.append("a b")
    wm.append("c d e f")
    assert wm.to_prompt().split() == ["c", "d", "e", "f"]
