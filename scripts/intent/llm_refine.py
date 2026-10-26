"""
OpenAI-compatible LLM client for compile_refine (DeepSeek / OpenAI / Azure-style).

Env (any of):
  REFINE_LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY
  REFINE_LLM_BASE_URL  (default: https://api.deepseek.com/v1)
  REFINE_LLM_MODEL     (default: deepseek-chat)
  REFINE_LLM_DISABLED=1  → force off
"""
from __future__ import annotations

import json
import os
import re
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from refine_schema import normalize_refine, schema_for_llm_prompt

# Stable per-process session for OpenCode Go routing (x-opencode-session).
_PROCESS_SESSION = os.environ.get("OPENCODE_SESSION") or str(uuid.uuid4())


def _load_opencode_auth_key() -> Optional[str]:
    """Read key from ``opencode auth login -p opencode`` → auth.json."""
    candidates = [
        Path.home() / ".local" / "share" / "opencode" / "auth.json",
        Path(os.environ.get("APPDATA") or "") / "opencode" / "auth.json",
        Path(os.environ.get("LOCALAPPDATA") or "") / "opencode" / "auth.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict):
            continue
        for provider in ("opencode", "opencode-go", "deepseek"):
            entry = data.get(provider)
            if isinstance(entry, dict) and entry.get("key"):
                return str(entry["key"]).strip()
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
    return None


def llm_available() -> bool:
    if os.environ.get("REFINE_LLM_DISABLED", "").strip() in ("1", "true", "yes"):
        return False
    return bool(
        os.environ.get("REFINE_LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _load_opencode_auth_key()
    )


def _cfg() -> Dict[str, str]:
    """Resolve API key. Prefer OpenCode auth.json over env (env often holds a stale Restricted key)."""
    file_key = (_load_opencode_auth_key() or "").strip()
    env_key = (
        os.environ.get("REFINE_LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    # Prefer file when present; allow env override only via REFINE_LLM_FORCE_ENV=1
    force_env = os.environ.get("REFINE_LLM_FORCE_ENV", "").strip() in ("1", "true", "yes")
    if force_env and env_key:
        key, from_opencode_file = env_key, False
    elif file_key:
        key, from_opencode_file = file_key, True
    else:
        key, from_opencode_file = env_key, False

    default_base = "https://api.deepseek.com/v1"
    if key.startswith("oc_sk_") or from_opencode_file:
        default_base = "https://opencode.ai/zen/go/v1"
    base = (
        os.environ.get("REFINE_LLM_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or default_base
    ).rstrip("/")
    default_model = "deepseek-chat"
    if "opencode.ai" in base:
        default_model = "deepseek-v4-flash"
    model = (
        os.environ.get("REFINE_LLM_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or default_model
    )
    return {
        "api_key": key,
        "base_url": base,
        "model": model,
        "key_source": "opencode_auth.json" if from_opencode_file else ("env" if key else "none"),
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


class OpenAICompatRefineClient:
    """Minimal chat.completions client; no third-party deps."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        cfg = _cfg()
        self.api_key = (api_key or cfg["api_key"]).strip()
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.model = model or cfg["model"]
        self.timeout_s = timeout_s
        self.key_source = cfg.get("key_source") or "unknown"

    def compile_refine(
        self,
        prompt: str,
        *,
        previous_action: Optional[str] = None,
        timeline: Optional[Dict[str, Any]] = None,
        previous_caption: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("no LLM API key")
        system = (
            "You compile motion-revise feedback into a closed Refine JSON schema.\n"
            "Use automatic_motion_caption / critique as observed evidence of the "
            "current clip; align refine ops with both user_feedback and that evidence.\n"
            + schema_for_llm_prompt()
        )
        user_parts = [
            f"previous_action: {previous_action or 'sit_down'}",
            f"user_feedback: {prompt}",
        ]
        if previous_caption:
            user_parts.append(
                "automatic_motion_caption: "
                + json.dumps(
                    {
                        "caption": previous_caption.get("caption"),
                        "critique": previous_caption.get("critique"),
                        "problems": previous_caption.get("problems"),
                        "source": previous_caption.get("source"),
                    },
                    ensure_ascii=False,
                )
            )
        if timeline:
            user_parts.append(
                "current_timeline: "
                + json.dumps(
                    {
                        "T": timeline.get("T"),
                        "phases": timeline.get("phases"),
                        "problems": timeline.get("problems"),
                        "sit_start": timeline.get("sit_start"),
                        "contact": timeline.get("contact"),
                        "hold_end": timeline.get("hold_end"),
                    },
                    ensure_ascii=False,
                )
            )
        body = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            "response_format": {"type": "json_object"},
        }
        # OpenCode Go requires x-opencode-session for API clients.
        # https://opencode.ai/docs/go/#where-can-i-use-it
        session_id = (
            os.environ.get("OPENCODE_SESSION")
            or os.environ.get("REFINE_LLM_SESSION")
            or _PROCESS_SESSION
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "BuildanAgent-refine/1.0",
            "Accept": "application/json",
            "x-opencode-session": session_id,
        }
        # Drop response_format for Go gateways that reject it.
        if "opencode.ai" in self.base_url:
            body.pop("response_format", None)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc

        content = (
            (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content")
            or ""
        )
        parsed = _extract_json(content)
        if not parsed:
            raise RuntimeError(f"LLM returned non-JSON: {content[:200]}")
        return normalize_refine(
            parsed,
            previous_action=previous_action,
            source="llm",
            raw_prompt=prompt,
        )


def get_default_llm_client() -> Optional[OpenAICompatRefineClient]:
    if not llm_available():
        return None
    return OpenAICompatRefineClient()
