"""Fan-out to multiple providers, then Kimi (Moonshot) summary when several succeed."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from parr_prompt.providers import (
    PROVIDERS,
    ProviderReply,
    complete_safe_multimodal,
    kimi_synthesize_safe,
)


def _dedupe_preserve_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _successful_replies(replies: list[ProviderReply]) -> list[ProviderReply]:
    return [
        r
        for r in replies
        if r.error is None and r.text is not None and r.text.strip()
    ]


def _format_synthesis_prompt(replies: list[ProviderReply]) -> str:
    lines = [
        "Below are answers from several AI systems (each ### block is one provider that succeeded).",
        "Output **only** a GitHub-flavored **Markdown table** (no text before or after the table).",
        "",
        "Columns:",
        "1. **Provider** — use the exact heading form `Label (provider_id)` from each block below "
        "(e.g. `OpenAI (openai)`, `Moonshot Kimi (kimi)`). One table row per block.",
        "2. **Short summary** — 2–4 sentences distilling that provider’s answer (main points only; not verbatim).",
        "",
        "Format: header row, then a separator row with pipes and dashes (e.g. | --- | --- |), then one data row per provider.",
        "",
        "---",
        "",
    ]
    for r in replies:
        lines.append(f"### {r.label} ({r.provider_id})")
        lines.append(r.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def ensemble_core(
    provider_ids: list[str],
    prompt: str,
    *,
    max_tokens: int | None = None,
    synth_model: str | None = None,
    synth_max_tokens: int | None = None,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
    images: list[tuple[bytes, str]] | None = None,
) -> tuple[list[ProviderReply], str | None, str | None]:
    """
    Run the same prompt on each provider in parallel. If exactly one successful answer, return it
    without calling Kimi. If two or more, ask Moonshot (Kimi) to summarize those answers only.
    Returns (replies, summary, error_message). summary is None when error_message is set.
    """
    ids = _dedupe_preserve_order(provider_ids)
    if not ids:
        return [], None, "No providers given for ensemble."

    replies_map: dict[str, ProviderReply] = {}

    def _one(pid: str) -> ProviderReply:
        return complete_safe_multimodal(
            pid,
            prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            images=images,
            max_tokens=max_tokens,
        )

    with ThreadPoolExecutor(max_workers=max(len(ids), 1)) as ex:
        futs = {ex.submit(_one, pid): pid for pid in ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                replies_map[pid] = fut.result()
            except Exception as e:
                spec = PROVIDERS[pid]
                replies_map[pid] = ProviderReply(
                    pid, spec.label, None, error=str(e)
                )

    replies = [replies_map[pid] for pid in ids]
    successful = _successful_replies(replies)
    if not successful:
        return (
            replies,
            None,
            "No successful provider responses; cannot synthesize.",
        )

    if len(successful) == 1:
        return replies, successful[0].text, None

    bundle = _format_synthesis_prompt(successful)
    summary, syn_err = kimi_synthesize_safe(
        bundle,
        model=synth_model,
        max_tokens=synth_max_tokens,
    )
    if syn_err:
        return replies, None, syn_err
    return replies, summary, None


def run_ensemble(
    provider_ids: list[str],
    prompt: str,
    *,
    max_tokens: int | None = None,
    synth_model: str | None = None,
    synth_max_tokens: int | None = None,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
    images: list[tuple[bytes, str]] | None = None,
) -> tuple[list[ProviderReply], str]:
    """
    CLI-oriented wrapper: exits the process on failure.
    """
    replies, summary, err = ensemble_core(
        provider_ids,
        prompt,
        max_tokens=max_tokens,
        synth_model=synth_model,
        synth_max_tokens=synth_max_tokens,
        image_bytes=image_bytes,
        mime_type=mime_type,
        images=images,
    )
    if err:
        print(err, file=sys.stderr)
        code = 3 if "successful provider" in err else 2
        sys.exit(code)
    assert summary is not None
    return replies, summary
