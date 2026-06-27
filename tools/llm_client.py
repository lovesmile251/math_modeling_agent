from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import currentframe
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    api_key: str | None
    base_url: str | None
    api_style: str
    enabled: bool
    timeout_seconds: float = 60.0
    retries: int = 2
    retry_backoff_seconds: float = 0.5
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    cache_enabled: bool = False
    cache_dir: str | None = None
    fast_mode: bool = False
    thinking: str | None = None
    reasoning_effort: str | None = None
    reason: str = ""

    @classmethod
    def from_env(cls, use_llm: bool) -> "LLMConfig":
        if not use_llm:
            return cls(
                provider="none",
                model=None,
                api_key=None,
                base_url=None,
                api_style="none",
                enabled=False,
                reason="LLM disabled.",
            )

        provider = os.environ.get("MMA_LLM_PROVIDER", "openai").lower()
        fast_mode = _env_bool("MMA_LLM_FAST_MODE", default=False)
        timeout_seconds = _env_float("MMA_LLM_TIMEOUT_SECONDS", 35.0 if fast_mode else 60.0, minimum=1.0)
        retries = _env_int("MMA_LLM_RETRIES", 1 if fast_mode else 2, minimum=0)
        retry_backoff_seconds = _env_float(
            "MMA_LLM_RETRY_BACKOFF_SECONDS",
            0.2 if fast_mode else 0.5,
            minimum=0.0,
        )
        temperature = _env_optional_float("MMA_LLM_TEMPERATURE", minimum=0.0)
        top_p = _env_optional_float("MMA_LLM_TOP_P", minimum=0.0, maximum=1.0)
        max_output_tokens = _env_optional_int("MMA_LLM_MAX_OUTPUT_TOKENS", minimum=1)
        if fast_mode and max_output_tokens is None:
            max_output_tokens = 2048
        cache_enabled = _env_bool("MMA_LLM_CACHE", default=fast_mode)
        cache_dir = os.environ.get("MMA_LLM_CACHE_DIR", ".cache/llm")
        if provider == "openai":
            return cls._from_provider_env(
                provider=provider,
                model_env="OPENAI_MODEL",
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                default_model=None,
                default_base_url=None,
                default_api_style="responses",
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
                fast_mode=fast_mode,
            )

        if provider == "deepseek":
            return cls._from_provider_env(
                provider=provider,
                model_env="DEEPSEEK_MODEL",
                api_key_env="DEEPSEEK_API_KEY",
                base_url_env="DEEPSEEK_BASE_URL",
                default_model="deepseek-v4-pro",
                default_base_url="https://api.deepseek.com",
                default_api_style="chat_completions",
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
                fast_mode=fast_mode,
                thinking=os.environ.get("DEEPSEEK_THINKING"),
                reasoning_effort=os.environ.get("DEEPSEEK_REASONING_EFFORT"),
            )

        if provider in {"openai_compatible", "openai-compatible", "compatible"}:
            return cls._from_provider_env(
                provider=provider,
                model_env="MMA_LLM_MODEL",
                api_key_env="MMA_LLM_API_KEY",
                base_url_env="MMA_LLM_BASE_URL",
                default_model=None,
                default_base_url=None,
                default_api_style="chat_completions",
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
                fast_mode=fast_mode,
                thinking=os.environ.get("MMA_LLM_THINKING"),
                reasoning_effort=os.environ.get("MMA_LLM_REASONING_EFFORT"),
            )

        return cls(
            provider=provider,
            model=None,
            api_key=None,
            base_url=None,
            api_style="none",
            enabled=False,
            reason="Unsupported LLM provider.",
        )

    @classmethod
    def _from_provider_env(
        cls,
        *,
        provider: str,
        model_env: str,
        api_key_env: str,
        base_url_env: str,
        default_model: str | None,
        default_base_url: str | None,
        default_api_style: str,
        timeout_seconds: float,
        retries: int,
        retry_backoff_seconds: float,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        cache_enabled: bool = False,
        cache_dir: str | None = None,
        fast_mode: bool = False,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> "LLMConfig":
        model = os.environ.get(model_env, default_model)
        api_key = os.environ.get(api_key_env)
        base_url = os.environ.get(base_url_env, default_base_url)
        api_style = os.environ.get("MMA_LLM_API_STYLE", default_api_style)

        if not api_key:
            return cls(
                provider=provider,
                model=model,
                api_key=None,
                base_url=base_url,
                api_style=api_style,
                enabled=False,
                reason=f"Missing {api_key_env}.",
            )
        if not model:
            return cls(
                provider=provider,
                model=None,
                api_key=api_key,
                base_url=base_url,
                api_style=api_style,
                enabled=False,
                reason=f"Missing {model_env}.",
            )
        if api_style == "chat_completions" and not base_url and provider != "openai":
            return cls(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=None,
                api_style=api_style,
                enabled=False,
                reason=f"Missing {base_url_env}.",
            )
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_style=api_style,
            enabled=True,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            fast_mode=fast_mode,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.last_call: dict[str, Any] = {}
        self.log_path: Path | None = None
        self._memory_cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def set_log_path(self, path: Path | str | None) -> None:
        self.log_path = Path(path) if path else None

    def complete(self, instructions: str, user_input: str) -> str:
        if not self.config.enabled:
            raise RuntimeError(self.config.reason or "LLM is not enabled.")

        last_error: Exception | None = None
        attempts_made = 0
        caller = _caller_label()
        model, model_source = self._model_for_caller(caller)
        cache_key = self._cache_key(instructions, user_input, model)
        cache_path = self._cache_path(cache_key, model)
        cached = self._memory_cache.get(cache_key)
        if cached is None and cache_path:
            cached = self._read_cache(cache_path)
        if cache_path:
            if cached is not None:
                self._remember_cache(cache_key, cached)
                self.last_call = {
                    "provider": self.config.provider,
                    "model": model or "",
                    "model_source": model_source,
                    "attempts": 0,
                    "ok": True,
                    "cache_hit": True,
                    "latency_seconds": 0.0,
                    "input_chars": len(instructions) + len(user_input),
                    "output_chars": len(cached),
                    "caller": caller,
                }
                self._append_call_log(self.last_call)
                return cached
        elif cached is not None:
            self.last_call = {
                "provider": self.config.provider,
                "model": model or "",
                "model_source": model_source,
                "attempts": 0,
                "ok": True,
                "cache_hit": True,
                "latency_seconds": 0.0,
                "input_chars": len(instructions) + len(user_input),
                "output_chars": len(cached),
                "caller": caller,
            }
            self._append_call_log(self.last_call)
            return cached
        for attempt in range(self.config.retries + 1):
            attempts_made = attempt + 1
            started = time.perf_counter()
            try:
                text = self._complete_once(instructions, user_input, model)
                self.last_call = {
                    "provider": self.config.provider,
                    "model": model or "",
                    "model_source": model_source,
                    "attempts": attempts_made,
                    "ok": True,
                    "cache_hit": False,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "input_chars": len(instructions) + len(user_input),
                    "output_chars": len(text),
                    "caller": caller,
                }
                self._remember_cache(cache_key, text)
                if cache_path:
                    self._write_cache(cache_path, text, model)
                self._append_call_log(self.last_call)
                return text
            except Exception as exc:
                last_error = exc
                if self._is_non_retryable(exc) or attempt >= self.config.retries:
                    break
                if self.config.retry_backoff_seconds:
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        self.last_call = {
            "provider": self.config.provider,
            "model": model or "",
            "model_source": model_source,
            "attempts": attempts_made,
            "ok": False,
            "cache_hit": False,
            "error": _redact_secrets(str(last_error)),
            "failure_kind": classify_llm_error(last_error),
            "caller": caller,
        }
        self._append_call_log(self.last_call)
        raise RuntimeError(f"LLM request failed after {self.config.retries + 1} attempt(s): {last_error}") from last_error

    def complete_json(
        self,
        instructions: str,
        user_input: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = parse_json_object(self.complete(instructions, user_input))
        if schema:
            _validate_json_schema_subset(payload, schema)
        return payload

    def _append_call_log(self, metadata: dict[str, Any]) -> None:
        if not self.log_path:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _model_for_caller(self, caller: str) -> tuple[str | None, str]:
        exact_env = _caller_model_env(caller)
        if exact_env:
            value = os.environ.get(exact_env)
            if value:
                return value, exact_env

        category_env = _caller_category_model_env(caller)
        if category_env:
            value = os.environ.get(category_env)
            if value:
                return value, category_env

        default_env = os.environ.get("MMA_LLM_MODEL_DEFAULT")
        if default_env:
            return default_env, "MMA_LLM_MODEL_DEFAULT"
        return self.config.model, "config"

    def _cache_key(self, instructions: str, user_input: str, model: str | None) -> str:
        payload = {
            "provider": self.config.provider,
            "api_style": self.config.api_style,
            "model": model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_output_tokens,
            "thinking": self.config.thinking,
            "reasoning_effort": self.config.reasoning_effort,
            "instructions": instructions,
            "user_input": user_input,
        }
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_path(self, key: str, model: str | None) -> Path | None:
        if not self.config.cache_enabled or not self.config.cache_dir:
            return None
        provider = _safe_path_part(self.config.provider)
        model_part = _safe_path_part(model or "unknown-model")
        return Path(self.config.cache_dir) / provider / model_part / f"{key}.json"

    def _remember_cache(self, key: str, text: str) -> None:
        if key in self._memory_cache:
            self._memory_cache[key] = text
            return
        if len(self._memory_cache) >= 128:
            self._memory_cache.pop(next(iter(self._memory_cache)))
        self._memory_cache[key] = text

    def _read_cache(self, path: Path) -> str | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) else None

    def _write_cache(self, path: Path, text: str, model: str | None) -> None:
        payload = {
            "provider": self.config.provider,
            "model": model,
            "api_style": self.config.api_style,
            "text": text,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return

    def _complete_once(self, instructions: str, user_input: str, model: str | None) -> str:
        if self.config.api_style == "responses":
            return self._complete_responses(instructions, user_input, model)
        if self.config.api_style == "chat_completions":
            return self._complete_chat_completions(instructions, user_input, model)
        raise RuntimeError(f"Unsupported LLM API style: {self.config.api_style}")

    def _is_non_retryable(self, exc: Exception) -> bool:
        status_code = _error_status_code(exc)
        if status_code is not None:
            retryable_client_errors = {408, 409, 425, 429}
            return 400 <= status_code < 500 and status_code not in retryable_client_errors

        text = str(exc)
        return (
            "Missing " in text
            or "Unsupported LLM provider" in text
            or "Unsupported LLM API style" in text
        )

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAI(**kwargs)

    def _complete_responses(self, instructions: str, user_input: str, model: str | None) -> str:
        client = self._openai_client()
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
            **self._request_controls(responses_api=True),
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise RuntimeError("LLM response did not contain output_text.")
        return text.strip()

    def _complete_chat_completions(self, instructions: str, user_input: str, model: str | None) -> str:
        client = self._openai_client()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            **self._request_controls(responses_api=False),
        }
        extra_body: dict[str, Any] = {}
        if self.config.reasoning_effort:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        if self.config.thinking:
            extra_body["thinking"] = {"type": self.config.thinking}
        if extra_body:
            payload["extra_body"] = extra_body

        response = client.chat.completions.create(**payload)
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("Chat completions response contained no choices array.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("Chat completions response choices[0] had no message object.")
        content = getattr(message, "content", "")
        if not content:
            raise RuntimeError("Chat completions response did not contain message content.")
        return str(content).strip()

    def _request_controls(self, *, responses_api: bool) -> dict[str, Any]:
        controls: dict[str, Any] = {}
        if self.config.temperature is not None:
            controls["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            controls["top_p"] = self.config.top_p
        if self.config.max_output_tokens is not None:
            key = "max_output_tokens" if responses_api else "max_tokens"
            controls[key] = self.config.max_output_tokens
        return controls


def build_llm_client(use_llm: bool) -> LLMClient:
    return LLMClient(LLMConfig.from_env(use_llm))


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object.")
    return data


def _validate_json_schema_subset(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    expected_type = schema.get("type")
    if expected_type and expected_type != "object":
        raise ValueError("Only object JSON schemas are supported.")
    for key in schema.get("required", []):
        if key not in payload:
            raise ValueError(f"LLM response JSON missing required key: {key}")


def _caller_label() -> str:
    frame = currentframe()
    if frame is None or frame.f_back is None:
        return ""
    caller = frame.f_back
    while caller and Path(caller.f_code.co_filename).name == "llm_client.py":
        caller = caller.f_back
    if caller is None:
        return ""
    return f"{Path(caller.f_code.co_filename).name}:{caller.f_code.co_name}"


def _caller_model_env(caller: str) -> str:
    filename = caller.split(":", 1)[0]
    if not filename.endswith(".py"):
        return ""
    stem = Path(filename).stem.upper()
    stem = re.sub(r"[^A-Z0-9]+", "_", stem)
    return f"MMA_LLM_MODEL_{stem}"


def _caller_category_model_env(caller: str) -> str:
    filename = caller.split(":", 1)[0]
    mapping = {
        "problem_agent.py": "MMA_LLM_MODEL_REASONING",
        "model_selection_crew.py": "MMA_LLM_MODEL_REASONING",
        "modeling_agent.py": "MMA_LLM_MODEL_REASONING",
        "modeling_critic_agent.py": "MMA_LLM_MODEL_REASONING",
        "decision_agent.py": "MMA_LLM_MODEL_REASONING",
        "experiment_plan_agent.py": "MMA_LLM_MODEL_PLANNING",
        "code_plan_agent.py": "MMA_LLM_MODEL_PLANNING",
        "writing_agent.py": "MMA_LLM_MODEL_WRITING",
        "math_reviewer.py": "MMA_LLM_MODEL_REVIEW",
        "fact_reviewer.py": "MMA_LLM_MODEL_REVIEW",
        "structure_reviewer.py": "MMA_LLM_MODEL_REVIEW",
        "language_reviewer.py": "MMA_LLM_MODEL_REVIEW",
        "review_agent.py": "MMA_LLM_MODEL_REVIEW",
        "code_repair_agent.py": "MMA_LLM_MODEL_CODE_REPAIR",
    }
    return mapping.get(filename, "")


def _error_status_code(exc: Exception) -> int | None:
    candidates: list[Any] = [exc]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        candidates.append(cause)
    context = getattr(exc, "__context__", None)
    if context is not None:
        candidates.append(context)

    for item in candidates:
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(item, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status

    match = re.search(r"\b(?:status|error|HTTP)\s*(?:code)?\s*[:=]?\s*(\d{3})\b", str(exc), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def classify_llm_error(exc: Exception | str | None) -> str:
    if exc is None:
        return ""
    status_code = _error_status_code(exc) if isinstance(exc, Exception) else None
    text = str(exc).lower()
    if status_code == 402 or "insufficient balance" in text or "billing" in text:
        return "quota"
    if status_code in {401, 403} or "unauthorized" in text or "invalid api key" in text:
        return "auth"
    if status_code == 429 or "rate limit" in text:
        return "rate_limit"
    if status_code is not None and 400 <= status_code < 500:
        return "client"
    return ""


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def _redact_secrets(text: str) -> str:
    redacted = text
    for name, value in os.environ.items():
        if not value or len(value) < 6:
            continue
        upper_name = name.upper()
        if any(marker in upper_name for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted = redacted.replace(value, "<redacted>")
    redacted = re.sub(
        r"(?i)\bBearer\s+([A-Za-z0-9._~+/=-]{6,})",
        "Bearer <redacted>",
        redacted,
    )
    return redacted


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(minimum, value)
    return value


def _env_optional_float(
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(minimum, value)
    return value


def _env_optional_int(name: str, minimum: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if minimum is not None:
        value = max(minimum, value)
    return value
