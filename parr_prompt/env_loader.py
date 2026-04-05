"""Load `.env` when python-dotenv is installed (project dir + legacy parent + CWD)."""

from __future__ import annotations

from pathlib import Path


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_dotenv_files() -> list[Path]:
    """
    `.env` locations, in load order (later entries override in `load_env` / `read_dotenv_file`).

    Supports:
    - Flat layout: `pyproject.toml` next to this file → `.env` in the same folder.
    - Nested layout: package in `repo/parr_prompt/` → `.env` in `repo/.env`.
    """
    pkg = _package_dir()
    candidates = [pkg.parent / ".env", pkg / ".env"]
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            out.append(p)
    return out


def _parse_env_file(p: Path) -> dict[str, str]:
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def load_env() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    for i, root_env in enumerate(_repo_dotenv_files()):
        load_dotenv(root_env, override=(i > 0))

    cwd_env = find_dotenv(usecwd=True)
    if cwd_env:
        load_dotenv(cwd_env, override=True)


def read_dotenv_file(path: Path | None = None) -> dict[str, str]:
    """
    Parse `.env` into raw KEY -> value (no variable expansion).
    Used to see which keys are declared in the file vs process environment only.
    """
    if path is not None:
        return _parse_env_file(path)
    merged: dict[str, str] = {}
    for p in _repo_dotenv_files():
        merged.update(_parse_env_file(p))
    return merged
