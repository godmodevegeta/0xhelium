"""OpenRouter natural-language scenario compiler.

POST https://openrouter.ai/api/v1/chat/completions with a strict `json_schema` response format and
`provider.require_parameters = true`, so the request is only routed to endpoints that support the
structured-output parameter. The returned JSON is then validated independently by
`robustlab.scenarios.schema.validate`, exactly like manual input: out-of-range values are
rejected with field errors (never clamped), and timing quantization is disclosed.

The API key is read from `OPENROUTER_API_KEY` and is only placed in the Authorization header; it is
never logged, stored or returned. Without a key the compiler reports `configured: false` and
refuses to compile; it never fabricates a response.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import httpx

from robustlab.llm.prompt import (
    MAX_PROMPT_CHARS,
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    to_scenario,
)
from robustlab.scenarios.schema import ScenarioError, validate

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
TIMEOUT_S = 30.0
MAX_TOKENS = 500


class CompileError(Exception):
    """A compile request that produced no usable model answer."""

    def __init__(self, code: str, message: str, status: int, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **self.detail}


class RateLimiter:
    """Sliding-window limits: `per_client` requests per client and `total` overall per window."""

    def __init__(self, per_client: int = 6, total: int = 20, window_s: float = 60.0,
                 clock=time.monotonic) -> None:  # fmt: skip
        self.per_client, self.total, self.window_s, self.clock = per_client, total, window_s, clock
        self._all: deque[float] = deque()
        self._clients: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """Record a request; return None if allowed, else seconds until a slot frees."""
        now = self.clock()
        with self._lock:
            for q in (self._all, *self._clients.values()):
                while q and now - q[0] >= self.window_s:
                    q.popleft()
            mine = self._clients.setdefault(client, deque())
            if len(mine) >= self.per_client:
                return self.window_s - (now - mine[0])
            if len(self._all) >= self.total:
                return self.window_s - (now - self._all[0])
            mine.append(now)
            self._all.append(now)
            return None


@dataclass
class OpenRouterCompiler:
    api_key: str | None
    model: str = DEFAULT_MODEL
    timeout_s: float = TIMEOUT_S
    max_tokens: int = MAX_TOKENS
    transport: httpx.BaseTransport | None = None  # injected in tests
    limiter: RateLimiter | None = None

    @classmethod
    def from_env(cls, **kwargs) -> OpenRouterCompiler:
        key = os.environ.get("OPENROUTER_API_KEY") or None
        model = os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        return cls(api_key=key, model=model, limiter=kwargs.pop("limiter", RateLimiter()), **kwargs)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "provider": "openrouter",
            "model": self.model,
            "model_source": "OPENROUTER_MODEL" if os.environ.get("OPENROUTER_MODEL") else "default",
            "structured_output": "response_format json_schema (strict), provider.require_parameters",
            "limits": {
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "timeout_s": self.timeout_s,
                "max_tokens": self.max_tokens,
                "rate": None
                if self.limiter is None
                else {
                    "per_client_per_min": self.limiter.per_client,
                    "total_per_min": self.limiter.total,
                },
            },
            "prompt_version": PROMPT_VERSION,
        }

    def request_body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robustlab_scenario",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
            "provider": {"require_parameters": True},
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }

    def compile(self, prompt: str, *, client: str = "local", command_ranges=None) -> dict:
        """Compile `prompt` into a validated scenario preview, or raise CompileError."""
        if not self.configured:
            raise CompileError("not_configured", "OPENROUTER_API_KEY is not set on the server", 503)
        if not isinstance(prompt, str) or not prompt.strip():
            raise CompileError("empty_prompt", "describe the scenario in words", 422)
        if len(prompt) > MAX_PROMPT_CHARS:
            raise CompileError("prompt_too_long", f"prompt exceeds {MAX_PROMPT_CHARS} characters",
                               422, {"length": len(prompt)})  # fmt: skip
        if self.limiter is not None:
            wait = self.limiter.check(client)
            if wait is not None:
                raise CompileError("rate_limited", "too many compile requests; try again shortly",
                                   429, {"retry_after_s": round(wait, 1)})  # fmt: skip
        started = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Robustness Studio",
        }
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as http:
                response = http.post(ENDPOINT, headers=headers, json=self.request_body(prompt))
        except httpx.TimeoutException as exc:
            raise CompileError("upstream_timeout", f"OpenRouter did not answer within "
                               f"{self.timeout_s:g} s", 504) from exc  # fmt: skip
        except httpx.HTTPError as exc:
            raise CompileError("upstream_unreachable", f"OpenRouter request failed: "
                               f"{type(exc).__name__}", 502) from exc  # fmt: skip
        latency = time.perf_counter() - started
        if response.status_code == 429:
            raise CompileError("upstream_rate_limited", "OpenRouter rate limit reached", 429)
        if response.status_code in (401, 403):
            raise CompileError("upstream_auth", "OpenRouter rejected the API key", 502)
        if response.status_code >= 400:
            raise CompileError("upstream_error", f"OpenRouter returned HTTP {response.status_code}",
                               502, {"upstream_status": response.status_code,
                                     "upstream_message": _error_message(response)})  # fmt: skip
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise CompileError(
                "invalid_model_output", "OpenRouter response had no message", 502
            ) from exc
        meta = {
            "provider": "openrouter",
            "requested_model": self.model,
            "model": body.get("model"),
            "upstream_provider": body.get("provider"),
            "usage": body.get("usage"),
            "latency_s": round(latency, 3),
            "finish_reason": choice.get("finish_reason"),
            "prompt_version": PROMPT_VERSION,
        }
        if message.get("refusal"):
            raise CompileError("model_refused", "the model declined the request", 502,
                               {"refusal": str(message["refusal"])[:300], "llm": meta})  # fmt: skip
        content = message.get("content")
        try:
            proposal = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            proposal = None
        if not isinstance(proposal, dict):
            raise CompileError("invalid_model_output", "the model did not return a JSON object",
                               502, {"excerpt": str(content)[:300], "llm": meta})  # fmt: skip
        extra = sorted(set(proposal) - set(RESPONSE_SCHEMA["properties"]))
        scenario_doc = to_scenario(proposal)
        result = {"proposal": proposal, "scenario_document": scenario_doc, "llm": meta,
                  "unexpected_fields": extra}  # fmt: skip
        try:
            kwargs = {} if command_ranges is None else {"command_ranges": command_ranges}
            if extra:
                raise ScenarioError(
                    [{"field": f, "reason": "not in the response schema"} for f in extra]
                )
            validated = validate(scenario_doc, **kwargs)
        except ScenarioError as exc:
            result.update(valid=False, errors=exc.errors)
            return result
        result.update(valid=True, errors=[], **validated.to_dict())
        return result


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", ""))[:300]
    except ValueError:
        return response.text[:300]
