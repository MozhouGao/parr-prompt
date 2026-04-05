"""Provider configs and completion calls."""

from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

from anthropic import Anthropic
from google import genai as google_genai
from google.genai import types as genai_types
from openai import OpenAI

# Providers we implement with image+text in this project (OpenAI-style or native multimodal).
# DeepSeek’s /chat/completions body only allows text in message content (no `image_url` parts).
VISION_CAPABLE: frozenset[str] = frozenset({"openai", "gemini", "claude", "grok", "kimi"})

@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    env_var: str
    default_model: str | None
    base_url: str | None = None


@dataclass
class ProviderReply:
    """Result of one provider call (success, skip, or API error)."""

    provider_id: str
    label: str
    text: str | None
    error: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai",
        "OpenAI",
        "OPENAI_API_KEY",
        "gpt-5.4-mini",
        None,
    ),
    "gemini": ProviderSpec(
        "gemini",
        "Google Gemini",
        "GOOGLE_API_KEY",
        "gemini-2.5-flash",
        None,
    ),
    "grok": ProviderSpec(
        "grok",
        "xAI Grok",
        "XAI_API_KEY",
        # Text: Grok 4 fast non-reasoning. With an image, `complete_safe_multimodal` switches to reasoning unless `-m` / XAI_IMAGE_MODEL.
        "grok-4-1-fast-non-reasoning",
        "https://api.x.ai/v1",
    ),
    "claude": ProviderSpec(
        "claude",
        "Anthropic Claude",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-20250514",
        None,
    ),
    "deepseek": ProviderSpec(
        "deepseek",
        "DeepSeek",
        "DEEPSEEK_API_KEY",
        # API id `deepseek-chat` = DeepSeek-V3.2 non-thinking; `deepseek-reasoner` = thinking (text-only API).
        "deepseek-chat",
        "https://api.deepseek.com",
    ),
    "kimi": ProviderSpec(
        "kimi",
        "Moonshot Kimi",
        "MOONSHOT_API_KEY",
        # Kimi K2.5 handles text and images on one model id (OpenAI-style image_url parts).
        "kimi-k2.5",
        # Default host; get_runner / synthesis use _moonshot_api_base() so MOONSHOT_BASE_URL overrides.
        "https://api.moonshot.ai/v1",
    ),
    "doubao": ProviderSpec(
        "doubao",
        "Doubao (Volcengine Ark)",
        "ARK_API_KEY",
        None,
        "https://ark.cn-beijing.volces.com/api/v3",
    ),
}

# Grok multimodal default (must match `complete_safe_multimodal` image branch).
GROK_DEFAULT_IMAGE_MODEL = "grok-4-1-fast-reasoning"


def _ui_model_caption(provider_id: str, api_model_id: str) -> str:
    """Web UI only: friendly labels. API requests still use `api_model_id` from `_resolve_model`."""
    if provider_id == "deepseek":
        if api_model_id == "deepseek-chat":
            return "DeepSeek-v3.2"
        if api_model_id == "deepseek-reasoner":
            return "DeepSeek-v3.2 (reasoner)"
        return api_model_id
    if provider_id == "claude" and api_model_id.startswith("claude-sonnet-4-"):
        return "claude-sonnet-4"
    if provider_id == "claude" and api_model_id == "claude-sonnet-4":
        return "claude-sonnet-4"
    return api_model_id


def provider_sidebar_model_caption(provider_id: str) -> str:
    """Effective default model id(s) for sidebar labels; honors the same env vars as `_resolve_model`."""
    spec = PROVIDERS[provider_id]
    if provider_id == "grok":
        text = _resolve_model("grok", None, spec) or ""
        img = os.environ.get("XAI_IMAGE_MODEL", "").strip() or GROK_DEFAULT_IMAGE_MODEL
        return (
            f"{_ui_model_caption('grok', text)} (text) · {_ui_model_caption('grok', img)} (image)"
        )
    m = _resolve_model(provider_id, None, spec)
    if m is None:
        return "set model in CLI (-m)"
    return _ui_model_caption(provider_id, m)


def _resolve_model(
    provider_id: str,
    model: str | None,
    spec: ProviderSpec,
) -> str | None:
    """Explicit `model`, env overrides, or `spec.default_model` (OpenAI/Grok/Kimi also read provider env when unset)."""
    if provider_id == "openai":
        if model and str(model).strip():
            return str(model).strip()
        env_m = os.environ.get("OPENAI_DEFAULT_MODEL", "").strip()
        if env_m:
            return env_m
        return spec.default_model
    if provider_id == "grok":
        if model and str(model).strip():
            return str(model).strip()
        env_m = os.environ.get("XAI_DEFAULT_MODEL", "").strip()
        if env_m:
            return env_m
        return spec.default_model
    if provider_id == "kimi":
        if model and str(model).strip():
            return str(model).strip()
        env_m = os.environ.get("MOONSHOT_DEFAULT_MODEL", "").strip() or os.environ.get(
            "MOONSHOT_VISION_MODEL", ""
        ).strip()
        if env_m:
            return env_m
        return spec.default_model
    if provider_id == "deepseek":
        if model and str(model).strip():
            return str(model).strip()
        env_t = os.environ.get("DEEPSEEK_DEFAULT_MODEL", "").strip()
        if env_t:
            return env_t
        return spec.default_model
    return model or spec.default_model


def _moonshot_api_base() -> str:
    """OpenAI-compatible Moonshot API host. Keys from platform.moonshot.ai need .ai; China console keys need .cn."""
    u = os.environ.get("MOONSHOT_BASE_URL", "").strip()
    if u:
        return u.rstrip("/")
    return "https://api.moonshot.ai/v1"


def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        print(f"Missing API key: set {env_var}", file=sys.stderr)
        sys.exit(2)
    return key


def _openai_compat_chat(
    api_key: str,
    base_url: str | None,
    model: str,
    prompt: str,
    max_tokens: int | None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content
    if content is None:
        return ""
    return content


def _openai_compat_chat_images(
    api_key: str,
    base_url: str | None,
    model: str,
    prompt: str,
    images: Sequence[tuple[bytes, str]],
    max_tokens: int | None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    user_content: list[dict] = [{"type": "text", "text": prompt}]
    for image_bytes, mime_type in images:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content
    if content is None:
        return ""
    return content


def _gemini_chat(api_key: str, model: str, prompt: str, max_tokens: int | None) -> str:
    client = google_genai.Client(api_key=api_key)
    config = None
    if max_tokens is not None:
        config = google_genai.types.GenerateContentConfig(max_output_tokens=max_tokens)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    if resp.text:
        return resp.text
    return ""


def _gemini_chat_images(
    api_key: str,
    model: str,
    prompt: str,
    images: Sequence[tuple[bytes, str]],
    max_tokens: int | None,
) -> str:
    client = google_genai.Client(api_key=api_key)
    config = None
    if max_tokens is not None:
        config = google_genai.types.GenerateContentConfig(max_output_tokens=max_tokens)
    parts: list = []
    for image_bytes, mime_type in images:
        parts.append(genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    parts.append(genai_types.Part.from_text(text=prompt))
    resp = client.models.generate_content(model=model, contents=parts, config=config)
    if resp.text:
        return resp.text
    return ""


def _claude_chat(api_key: str, model: str, prompt: str, max_tokens: int | None) -> str:
    client = Anthropic(api_key=api_key)
    mt = max_tokens if max_tokens is not None else 4096
    msg = client.messages.create(
        model=model,
        max_tokens=mt,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def _claude_chat_images(
    api_key: str,
    model: str,
    prompt: str,
    images: Sequence[tuple[bytes, str]],
    max_tokens: int | None,
) -> str:
    client = Anthropic(api_key=api_key)
    mt = max_tokens if max_tokens is not None else 4096
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for image_bytes, mime_type in images:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64,
                },
            }
        )
    msg = client.messages.create(
        model=model,
        max_tokens=mt,
        messages=[{"role": "user", "content": blocks}],
    )
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def get_runner(provider_id: str) -> Callable[..., str]:
    spec = PROVIDERS[provider_id]
    if provider_id == "gemini":

        def run(api_key: str, model: str, prompt: str, max_tokens: int | None) -> str:
            return _gemini_chat(api_key, model, prompt, max_tokens)

        return run
    if provider_id == "claude":

        def run(api_key: str, model: str, prompt: str, max_tokens: int | None) -> str:
            return _claude_chat(api_key, model, prompt, max_tokens)

        return run

    def run(api_key: str, model: str, prompt: str, max_tokens: int | None) -> str:
        base = _moonshot_api_base() if provider_id == "kimi" else spec.base_url
        return _openai_compat_chat(api_key, base, model, prompt, max_tokens)

    return run


def _run_multimodal(
    provider_id: str,
    api_key: str,
    model: str,
    prompt: str,
    images: Sequence[tuple[bytes, str]],
    max_tokens: int | None,
) -> str:
    if not images:
        raise ValueError("multimodal requires at least one image")
    spec = PROVIDERS[provider_id]
    if provider_id in ("openai", "grok"):
        return _openai_compat_chat_images(
            api_key, spec.base_url, model, prompt, images, max_tokens
        )
    if provider_id == "kimi":
        return _openai_compat_chat_images(
            api_key,
            _moonshot_api_base(),
            model,
            prompt,
            images,
            max_tokens,
        )
    if provider_id == "gemini":
        return _gemini_chat_images(api_key, model, prompt, images, max_tokens)
    if provider_id == "claude":
        return _claude_chat_images(api_key, model, prompt, images, max_tokens)
    raise ValueError(f"Unhandled multimodal provider: {provider_id}")


def complete_safe_multimodal(
    provider_id: str,
    prompt: str,
    *,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
    images: Sequence[tuple[bytes, str]] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ProviderReply:
    """Text-only or image+text (one or many images). Non-vision providers skip when images are set."""
    imgs: list[tuple[bytes, str]] = []
    if images:
        imgs = [(b, m or "image/png") for b, m in images if b]
    elif image_bytes:
        imgs = [(image_bytes, mime_type or "image/png")]
    if not imgs:
        return complete_safe(provider_id, prompt, model=model, max_tokens=max_tokens)
    if provider_id not in VISION_CAPABLE:
        spec = PROVIDERS[provider_id]
        return ProviderReply(
            provider_id,
            spec.label,
            None,
            error=(
                "skipped: image not supported — DeepSeek’s API only accepts text in messages "
                "(no image_url parts); Doubao has no image path in this app"
            ),
        )
    spec = PROVIDERS[provider_id]
    key = os.environ.get(spec.env_var, "").strip()
    if not key:
        return ProviderReply(
            provider_id,
            spec.label,
            None,
            error=f"skipped: missing {spec.env_var}",
        )
    resolved = _resolve_model(provider_id, model, spec)
    if provider_id == "grok" and imgs and not (model and str(model).strip()):
        resolved = os.environ.get("XAI_IMAGE_MODEL", "").strip() or GROK_DEFAULT_IMAGE_MODEL
    if not resolved:
        return ProviderReply(
            provider_id,
            spec.label,
            None,
            error="skipped: no default model",
        )
    try:
        text = _run_multimodal(provider_id, key, resolved, prompt, imgs, max_tokens)
        return ProviderReply(provider_id, spec.label, text, None)
    except Exception as e:
        return ProviderReply(provider_id, spec.label, None, error=str(e))


def complete_safe(
    provider_id: str,
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ProviderReply:
    """Call one provider; never exits the process. Missing key or errors → ProviderReply.error."""
    spec = PROVIDERS[provider_id]
    key = os.environ.get(spec.env_var, "").strip()
    if not key:
        return ProviderReply(
            provider_id,
            spec.label,
            None,
            error=f"skipped: missing {spec.env_var}",
        )
    resolved = _resolve_model(provider_id, model, spec)
    if not resolved:
        return ProviderReply(
            provider_id,
            spec.label,
            None,
            error="skipped: no default model (set a model in provider config or extend CLI)",
        )
    try:
        runner = get_runner(provider_id)
        text = runner(key, resolved, prompt, max_tokens)
        return ProviderReply(provider_id, spec.label, text, None)
    except Exception as e:
        return ProviderReply(provider_id, spec.label, None, error=str(e))


def complete(
    provider_id: str,
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    spec = PROVIDERS[provider_id]
    api_key = _require_key(spec.env_var)
    resolved_model = _resolve_model(provider_id, model, spec)
    if not resolved_model:
        print(
            f"Provider {provider_id} has no default model; pass --model <id>",
            file=sys.stderr,
        )
        sys.exit(2)
    runner = get_runner(provider_id)
    return runner(api_key, resolved_model, prompt, max_tokens)


def gemini_synthesize_safe(
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str | None, str | None]:
    """Returns (text, error). Never exits."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return None, "Missing GOOGLE_API_KEY"
    m = model or PROVIDERS["gemini"].default_model or "gemini-2.5-flash"
    try:
        return _gemini_chat(key, m, user_prompt, max_tokens), None
    except Exception as e:
        return None, str(e)


def kimi_synthesize_safe(
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str | None, str | None]:
    """Merge/summarize via Moonshot (Kimi) chat API. Returns (text, error). Never exits."""
    spec = PROVIDERS["kimi"]
    key = os.environ.get(spec.env_var, "").strip()
    if not key:
        return None, "Missing MOONSHOT_API_KEY"
    m = _resolve_model("kimi", model, spec)
    if not m:
        return None, "No Kimi model configured for synthesis"
    try:
        text = _openai_compat_chat(key, _moonshot_api_base(), m, user_prompt, max_tokens)
        return text, None
    except Exception as e:
        return None, str(e)


def gemini_synthesize(
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run Gemini with an already-built prompt. Exits if key missing."""
    api_key = _require_key("GOOGLE_API_KEY")
    m = model or PROVIDERS["gemini"].default_model or "gemini-2.5-flash"
    return _gemini_chat(api_key, m, user_prompt, max_tokens)
