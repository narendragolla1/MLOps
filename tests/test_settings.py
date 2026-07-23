import pytest

from omniai.settings import OmniSettings


def test_defaults_are_dev_friendly():
    s = OmniSettings(_env_file=None)
    assert s.database_url.startswith("sqlite+aiosqlite")
    assert s.auth_enabled  # auth on by default (fail closed)
    assert s.engine_managed


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("OMNIAI_DATABASE_URL", "postgresql+asyncpg://u:p@db/omni")
    monkeypatch.setenv("OMNIAI_API_KEYS", "k1, k2 ,k3")
    monkeypatch.setenv("OMNIAI_CORS_ORIGINS", "https://a.com,https://b.com")
    monkeypatch.setenv("OMNIAI_RATE_LIMIT_RPS", "2.5")
    monkeypatch.setenv("OMNIAI_ENGINE_BASE_URL", "http://vllm:8000")
    monkeypatch.setenv("OMNIAI_ENGINE_MANAGED", "false")
    s = OmniSettings(_env_file=None)
    assert s.database_url.startswith("postgresql+asyncpg")
    assert s.api_keys == ["k1", "k2", "k3"]
    assert s.cors_origins == ["https://a.com", "https://b.com"]
    assert s.rate_limit_rps == 2.5
    assert s.engine_base_url == "http://vllm:8000"
    assert not s.engine_managed


def test_fail_closed_without_keys():
    s = OmniSettings(_env_file=None)
    with pytest.raises(RuntimeError, match="OMNIAI_API_KEYS"):
        s.validate_security()


def test_explicit_auth_opt_out():
    OmniSettings(_env_file=None, auth_disabled=True).validate_security()  # no raise


def test_keys_satisfy_validation():
    OmniSettings(_env_file=None, api_keys=["k"]).validate_security()  # no raise
