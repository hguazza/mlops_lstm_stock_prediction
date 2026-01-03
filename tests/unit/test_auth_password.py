"""Unit tests for password hashing and verification."""

from src.domain.auth.password import hash_password, verify_password


class TestPasswordHashing:
    """Test password hashing functionality."""

    def test_hash_password_creates_hash(self):
        """Test that hash_password creates a hash."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password
        assert len(hashed) > 0

    def test_hash_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "test_password_123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password("", hashed) is False

    def test_hash_empty_password(self):
        """Test hashing empty password."""
        hashed = hash_password("")

        assert hashed is not None
        assert isinstance(hashed, str)
