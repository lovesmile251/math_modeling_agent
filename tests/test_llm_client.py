from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tools.llm_client as llm_client_module
from tools.llm_client import LLMClient, LLMConfig


def test_llm_config_uses_safe_numeric_env_defaults(monkeypatch):
    monkeypatch.setenv("MMA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    monkeypatch.setenv("MMA_LLM_TIMEOUT_SECONDS", "bad")
    monkeypatch.setenv("MMA_LLM_RETRIES", "-3")
    monkeypatch.setenv("MMA_LLM_RETRY_BACKOFF_SECONDS", "-1")

    config = LLMConfig.from_env(use_llm=True)

    assert config.enabled is True
    assert config.api_style == "responses"
    assert config.timeout_seconds == 60.0
    assert config.retries == 0
    assert config.retry_backoff_seconds == 0.0


def test_llm_config_reads_request_control_env(monkeypatch):
    monkeypatch.setenv("MMA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    monkeypatch.setenv("MMA_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("MMA_LLM_TOP_P", "1.5")
    monkeypatch.setenv("MMA_LLM_MAX_OUTPUT_TOKENS", "2048")

    config = LLMConfig.from_env(use_llm=True)

    assert config.temperature == 0.2
    assert config.top_p == 1.0
    assert config.max_output_tokens == 2048


def test_llm_fast_mode_sets_speed_defaults(monkeypatch):
    monkeypatch.setenv("MMA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    monkeypatch.setenv("MMA_LLM_FAST_MODE", "1")
    monkeypatch.delenv("MMA_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MMA_LLM_RETRIES", raising=False)
    monkeypatch.delenv("MMA_LLM_RETRY_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("MMA_LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("MMA_LLM_CACHE", raising=False)

    config = LLMConfig.from_env(use_llm=True)

    assert config.fast_mode is True
    assert config.timeout_seconds == 35.0
    assert config.retries == 1
    assert config.retry_backoff_seconds == 0.2
    assert config.max_output_tokens == 2048
    assert config.cache_enabled is True


def test_deepseek_config_uses_openai_compatible_chat_completions(monkeypatch):
    monkeypatch.setenv("MMA_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    config = LLMConfig.from_env(use_llm=True)

    assert config.enabled is True
    assert config.model == "deepseek-v4-pro"
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_style == "chat_completions"


def test_openai_compatible_config_requires_generic_model_and_base_url(monkeypatch):
    monkeypatch.setenv("MMA_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MMA_LLM_API_KEY", "key")
    monkeypatch.setenv("MMA_LLM_MODEL", "provider-model")
    monkeypatch.setenv("MMA_LLM_BASE_URL", "https://example.test/v1")

    config = LLMConfig.from_env(use_llm=True)

    assert config.enabled is True
    assert config.provider == "openai_compatible"
    assert config.model == "provider-model"
    assert config.base_url == "https://example.test/v1"
    assert config.api_style == "chat_completions"


def test_llm_client_records_success_metadata():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)
    client._complete_once = lambda instructions, user_input, model: "ok"  # type: ignore[method-assign]

    assert client.complete("system", "user") == "ok"
    assert client.last_call["ok"] is True
    assert client.last_call["attempts"] == 1
    assert client.last_call["input_chars"] == len("systemuser")
    assert client.last_call["output_chars"] == 2
    assert client.last_call["model"] == "unit-test"
    assert client.last_call["model_source"] == "config"


def test_llm_client_writes_call_metadata_log_without_prompt_text(tmp_path):
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    log_path = tmp_path / "logs" / "llm_calls.jsonl"
    client = LLMClient(config)
    client.set_log_path(log_path)
    client._complete_once = lambda _instructions, _user_input, _model: "ok"  # type: ignore[method-assign]

    client.complete("secret system prompt", "private user input")

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["provider"] == "openai"
    assert record["model"] == "unit-test"
    assert record["input_chars"] == len("secret system promptprivate user input")
    assert "secret system prompt" not in log_path.read_text(encoding="utf-8")
    assert "private user input" not in log_path.read_text(encoding="utf-8")


def test_llm_client_uses_response_cache(tmp_path):
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
        cache_enabled=True,
        cache_dir=str(tmp_path / "llm-cache"),
    )
    client = LLMClient(config)
    calls = []

    def complete_once(_instructions: str, _user_input: str, _model: str | None) -> str:
        calls.append("called")
        return "cached text"

    client._complete_once = complete_once  # type: ignore[method-assign]

    assert client.complete("secret system prompt", "private user input") == "cached text"
    assert client.last_call["cache_hit"] is False
    assert client.complete("secret system prompt", "private user input") == "cached text"
    assert client.last_call["cache_hit"] is True
    assert client.last_call["attempts"] == 0
    assert calls == ["called"]

    cache_text = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "llm-cache").rglob("*.json"))
    assert "cached text" in cache_text
    assert "secret system prompt" not in cache_text
    assert "private user input" not in cache_text


def test_llm_client_uses_in_memory_cache_without_disk_cache():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
        cache_enabled=False,
    )
    client = LLMClient(config)
    calls = []

    def complete_once(_instructions: str, _user_input: str, _model: str | None) -> str:
        calls.append("called")
        return "memory cached text"

    client._complete_once = complete_once  # type: ignore[method-assign]

    assert client.complete("system", "user") == "memory cached text"
    assert client.complete("system", "user") == "memory cached text"
    assert client.last_call["cache_hit"] is True
    assert client.last_call["attempts"] == 0
    assert calls == ["called"]


def test_llm_client_records_failure_metadata_without_sleep():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=1,
        retry_backoff_seconds=0,
    )
    client = LLMClient(config)

    def fail(_instructions: str, _user_input: str, _model: str | None) -> str:
        raise RuntimeError("temporary failure")

    client._complete_once = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="LLM request failed"):
        client.complete("system", "user")
    assert client.last_call["ok"] is False
    assert client.last_call["attempts"] == 2
    assert "temporary failure" in client.last_call["error"]


def test_llm_client_redacts_secrets_in_failure_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="sk-test-secret-value",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
        retry_backoff_seconds=0,
    )
    log_path = tmp_path / "llm_calls.jsonl"
    client = LLMClient(config)
    client.set_log_path(log_path)

    def fail(_instructions: str, _user_input: str, _model: str | None) -> str:
        raise RuntimeError("bad key sk-test-secret-value via Bearer abcdef123456")

    client._complete_once = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="LLM request failed"):
        client.complete("system", "user")

    log_text = log_path.read_text(encoding="utf-8")
    assert "sk-test-secret-value" not in client.last_call["error"]
    assert "abcdef123456" not in client.last_call["error"]
    assert "sk-test-secret-value" not in log_text
    assert "abcdef123456" not in log_text
    assert "<redacted>" in client.last_call["error"]


def test_llm_client_records_actual_attempts_for_non_retryable_failure():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=3,
        retry_backoff_seconds=0,
    )
    client = LLMClient(config)

    def fail(_instructions: str, _user_input: str, _model: str | None) -> str:
        raise RuntimeError("Unsupported LLM provider: test")

    client._complete_once = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="LLM request failed"):
        client.complete("system", "user")
    assert client.last_call["ok"] is False
    assert client.last_call["attempts"] == 1


def test_llm_client_does_not_retry_non_retryable_http_status():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=3,
        retry_backoff_seconds=0,
    )
    client = LLMClient(config)

    class StatusError(RuntimeError):
        status_code = 401

    def fail(_instructions: str, _user_input: str, _model: str | None) -> str:
        raise StatusError("unauthorized")

    client._complete_once = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="LLM request failed"):
        client.complete("system", "user")
    assert client.last_call["attempts"] == 1


def test_llm_client_retries_retryable_http_status():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=2,
        retry_backoff_seconds=0,
    )
    client = LLMClient(config)

    class StatusError(RuntimeError):
        status_code = 429

    def fail(_instructions: str, _user_input: str, _model: str | None) -> str:
        raise StatusError("rate limited")

    client._complete_once = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="LLM request failed"):
        client.complete("system", "user")
    assert client.last_call["attempts"] == 3


def test_llm_client_uses_category_model_override(monkeypatch):
    config = LLMConfig(
        provider="openai",
        model="default-model",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)
    used_models = []
    client._complete_once = (  # type: ignore[method-assign]
        lambda _instructions, _user_input, model: used_models.append(model) or "ok"
    )
    monkeypatch.setattr(llm_client_module, "_caller_label", lambda: "writing_agent.py:_draft")
    monkeypatch.setenv("MMA_LLM_MODEL_WRITING", "writing-model")

    client.complete("system", "user")

    assert used_models == ["writing-model"]
    assert client.last_call["model"] == "writing-model"
    assert client.last_call["model_source"] == "MMA_LLM_MODEL_WRITING"


def test_llm_client_exact_agent_model_override_wins(monkeypatch):
    config = LLMConfig(
        provider="openai",
        model="default-model",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)
    used_models = []
    client._complete_once = (  # type: ignore[method-assign]
        lambda _instructions, _user_input, model: used_models.append(model) or "ok"
    )
    monkeypatch.setattr(llm_client_module, "_caller_label", lambda: "writing_agent.py:_draft")
    monkeypatch.setenv("MMA_LLM_MODEL_WRITING", "writing-model")
    monkeypatch.setenv("MMA_LLM_MODEL_WRITING_AGENT", "exact-writing-model")

    client.complete("system", "user")

    assert used_models == ["exact-writing-model"]
    assert client.last_call["model_source"] == "MMA_LLM_MODEL_WRITING_AGENT"


def test_chat_completions_response_parsing():
    config = LLMConfig(
        provider="deepseek",
        model="unit-test",
        api_key="key",
        base_url="https://api.deepseek.com",
        api_style="chat_completions",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  chat ok  "))]
    )
    completions = SimpleNamespace(create=lambda **_payload: response)
    chat = SimpleNamespace(completions=completions)
    client._openai_client = lambda: SimpleNamespace(chat=chat)  # type: ignore[method-assign]

    assert client.complete("system", "user") == "chat ok"


def test_chat_completions_includes_request_controls():
    config = LLMConfig(
        provider="deepseek",
        model="unit-test",
        api_key="key",
        base_url="https://api.deepseek.com",
        api_style="chat_completions",
        enabled=True,
        retries=0,
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=512,
    )
    client = LLMClient(config)
    captured = {}

    def create(**payload):
        captured.update(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    client._openai_client = lambda: SimpleNamespace(chat=chat)  # type: ignore[method-assign]

    assert client.complete("system", "user") == "ok"
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 0.9
    assert captured["max_tokens"] == 512


def test_responses_includes_request_controls():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
        temperature=0.3,
        top_p=0.8,
        max_output_tokens=256,
    )
    client = LLMClient(config)
    captured = {}

    def create(**payload):
        captured.update(payload)
        return SimpleNamespace(output_text="ok")

    responses = SimpleNamespace(create=create)
    client._openai_client = lambda: SimpleNamespace(responses=responses)  # type: ignore[method-assign]

    assert client.complete("system", "user") == "ok"
    assert captured["temperature"] == 0.3
    assert captured["top_p"] == 0.8
    assert captured["max_output_tokens"] == 256


def test_complete_json_parses_fenced_object():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)
    client.complete = lambda _instructions, _user_input: '```json\n{"ok": true}\n```'  # type: ignore[method-assign]

    assert client.complete_json("system", "user") == {"ok": True}


def test_complete_json_validates_required_keys():
    config = LLMConfig(
        provider="openai",
        model="unit-test",
        api_key="key",
        base_url=None,
        api_style="responses",
        enabled=True,
        retries=0,
    )
    client = LLMClient(config)
    client.complete = lambda _instructions, _user_input: '{"ok": true}'  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="missing required key"):
        client.complete_json("system", "user", schema={"type": "object", "required": ["tasks"]})
