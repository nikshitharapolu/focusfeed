from app.auth import _hash_password, _verify_password

def test_password_hash_is_salted_and_verifiable():
    first = _hash_password("correct-horse")
    second = _hash_password("correct-horse")
    assert first != second
    assert "correct-horse" not in first
    assert _verify_password("correct-horse", first)
    assert not _verify_password("wrong-password", first)
