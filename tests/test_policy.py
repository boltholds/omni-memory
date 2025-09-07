# tests/test_policy.py
import time
from domain.policy import MemoryPolicy


def test_filter_write_blocks_email_and_api_key():
    pol = MemoryPolicy()

    ok = {"id": "1", "type": "note", "text": "just a story"}
    bad1 = {"id": "2", "type": "note", "text": "email me at root@example.com"}
    bad2 = {"id": "3", "type": "note", "text": "api_key=ABCDEF1234567890"}

    assert pol.filter_write(ok) is True
    assert pol.filter_write(bad1) is False
    assert pol.filter_write(bad2) is False


def test_apply_ttl_sets_different_expiry_for_volatility():
    pol = MemoryPolicy()
    obj_high = {"id": "x", "meta": {"volatility": "high"}}
    obj_norm = {"id": "y", "meta": {"volatility": "normal"}}

    ttl_high = pol.apply_ttl(obj_high)
    ttl_norm = pol.apply_ttl(obj_norm)

    # TTL должно быть в будущем
    now = time.time()
    assert ttl_high > now
    assert ttl_norm > now
    # high volatility TTL < normal TTL
    assert ttl_high < ttl_norm
