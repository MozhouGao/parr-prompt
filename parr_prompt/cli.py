"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parr_prompt.ensemble import run_ensemble
from parr_prompt.providers import PROVIDERS, complete


def _read_prompt(args: argparse.Namespace) -> str:
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Provide a prompt as an argument, use -f FILE, or pipe stdin.", file=sys.stderr)
    sys.exit(2)


def _parse_providers(raw: str) -> list[str]:
    ids = sorted(PROVIDERS.keys())
    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [p for p in parts if p not in PROVIDERS]
    if bad:
        print(f"Unknown provider(s): {', '.join(bad)}. Valid: {', '.join(ids)}", file=sys.stderr)
        sys.exit(2)
    return parts


def build_parser() -> argparse.ArgumentParser:
    ids = sorted(PROVIDERS.keys())
    p = argparse.ArgumentParser(
        prog="parr-prompt",
        description="Send a prompt to one GenAI provider, or fan out to several and summarize with Kimi (Moonshot).",
    )
    mux = p.add_mutually_exclusive_group(required=False)
    mux.add_argument(
        "-p",
        "--provider",
        choices=ids,
        metavar="NAME",
        help=f"Single provider: {', '.join(ids)}",
    )
    mux.add_argument(
        "--providers",
        metavar="LIST",
        help="Comma-separated providers; Kimi merges answers if 2+ succeed (needs MOONSHOT_API_KEY then)",
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Prompt text (optional if -f or stdin)",
    )
    p.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        help="Read prompt from file (UTF-8)",
    )
    p.add_argument(
        "-m",
        "--model",
        metavar="ID",
        help="With -p only: override that provider's model",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max output tokens per provider call (-p or --providers). Summary uses --synth-max-tokens.",
    )
    p.add_argument(
        "--synth-model",
        metavar="ID",
        help="Kimi (Moonshot) model for the final summary (default: kimi-k2.5)",
    )
    p.add_argument(
        "--synth-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max output tokens for the Kimi summary",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="With --providers: print each model's reply on stderr before the summary",
    )
    p.add_argument(
        "-l",
        "--list",
        action="store_true",
        dest="list_providers",
        help="List providers, env vars, and default models, then exit",
    )
    return p


def main() -> None:
    from parr_prompt.env_loader import load_env

    load_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.list_providers:
        for pid in sorted(PROVIDERS.keys()):
            s = PROVIDERS[pid]
            dm = s.default_model or "(no default — use --model)"
            base = f" base_url={s.base_url}" if s.base_url else ""
            print(f"{pid:10}  {s.env_var:20}  default_model={dm}{base}")
        return
    if not args.provider and not args.providers:
        parser.error("one of the following arguments is required: -p/--provider or --providers")

    text = _read_prompt(args)
    if not text.strip():
        print("Empty prompt.", file=sys.stderr)
        sys.exit(2)

    if args.providers:
        plist = _parse_providers(args.providers)
        if not plist:
            print("Empty --providers list.", file=sys.stderr)
            sys.exit(2)
        replies, summary = run_ensemble(
            plist,
            text,
            max_tokens=args.max_tokens,
            synth_model=args.synth_model,
            synth_max_tokens=args.synth_max_tokens,
        )
        for r in replies:
            if r.error:
                print(f"[{r.provider_id}] {r.error}", file=sys.stderr)
        if args.verbose:
            for r in replies:
                if r.text is None:
                    continue
                print(f"\n### {r.label} ({r.provider_id})", file=sys.stderr)
                print(r.text, file=sys.stderr)
            print("", file=sys.stderr)
        sys.stdout.write(summary)
        if summary and not summary.endswith("\n"):
            sys.stdout.write("\n")
        return

    out = complete(
        args.provider,
        text,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    sys.stdout.write(out)
    if out and not out.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
