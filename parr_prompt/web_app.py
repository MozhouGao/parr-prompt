"""Dash UI: Open WebUI–style shell (sidebar + chat + composer)."""

from __future__ import annotations

import base64
import os
from contextlib import contextmanager
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, callback, dcc, html

from parr_prompt.ensemble import ensemble_core
from parr_prompt.env_loader import read_dotenv_file
from parr_prompt.providers import (
    PROVIDERS,
    ProviderReply,
    complete_safe_multimodal,
    provider_sidebar_model_caption,
)

_ROOT = Path(__file__).resolve().parent


def _provider_key_configured(
    env_var: str,
    session_keys: dict | None,
    file_values: dict[str, str],
) -> bool:
    """True if the key is set in session UI, or non-empty in .env file, or in os.environ."""
    if (session_keys or {}).get(env_var, "").strip():
        return True
    if (file_values.get(env_var) or "").strip():
        return True
    if os.environ.get(env_var, "").strip():
        return True
    return False


def _ensemble_checklist_options(session_keys: dict | None) -> list[dict]:
    file_vals = read_dotenv_file()
    opts: list[dict] = []
    for pid, spec in sorted(PROVIDERS.items()):
        ok = _provider_key_configured(spec.env_var, session_keys, file_vals)
        model_line = provider_sidebar_model_caption(pid)
        name_line = spec.label + ("" if ok else " — add key below")
        opts.append(
            {
                "label": html.Div(
                    className="owui-checklist-option-label",
                    children=[
                        html.Div(name_line, className="owui-provider-name"),
                        html.Div(model_line, className="owui-provider-model"),
                    ],
                ),
                "value": pid,
                "disabled": not ok,
            }
        )
    return opts


def _default_ensemble_value(options: list[dict]) -> list[str]:
    allowed = [o["value"] for o in options if not o.get("disabled")]
    return [allowed[0]] if allowed else []


@contextmanager
def _api_env_session(session_keys: dict | None):
    """Temporarily overlay session API keys onto os.environ for provider calls."""
    if not session_keys:
        yield
        return
    patch = {k: str(v).strip() for k, v in session_keys.items() if v and str(v).strip()}
    if not patch:
        yield
        return
    saved: dict[str, str | None] = {}
    try:
        for k, v in patch.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENTS = 12


def _empty_state() -> html.Div:
    return html.Div(
        className="owui-empty",
        children=[
            html.H2("How can I help you today?"),
            html.P("Select GenAI providers that you have API keys."),
        ],
    )


def _parse_data_uri(contents: str | None) -> tuple[bytes | None, str | None]:
    if not contents or "," not in contents:
        return None, None
    meta, b64 = contents.split(",", 1)
    if ";" in meta and "base64" in meta:
        mime = meta.split(":", 1)[1].split(";", 1)[0]
    elif ":" in meta:
        mime = meta.split(":", 1)[1]
    else:
        mime = "image/png"
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None, None
    return raw, mime


def _reply_to_markdown(text: str | None) -> str:
    if not text:
        return "_No text returned._"
    return text


def _img_data_url(image_data: dict | None) -> str | None:
    if not image_data or not isinstance(image_data, dict) or not image_data.get("b64"):
        return None
    mime = image_data.get("mime") or "image/png"
    return f"data:{mime};base64,{image_data['b64']}"


def _store_image_items(image_store_data) -> list[dict]:
    """Normalize image-store payload to a list of {b64, mime} dicts."""
    if isinstance(image_store_data, list):
        return [x for x in image_store_data if isinstance(x, dict) and x.get("b64")]
    if isinstance(image_store_data, dict) and image_store_data.get("b64"):
        return [image_store_data]
    return []


def _decode_images_from_store(image_store_data) -> list[tuple[bytes, str]]:
    out: list[tuple[bytes, str]] = []
    for item in _store_image_items(image_store_data):
        try:
            raw = base64.b64decode(item["b64"])
            mime = item.get("mime") or "image/png"
            out.append((raw, mime))
        except Exception:
            continue
    return out


def _default_vision_prompt(n_images: int) -> str:
    if n_images <= 0:
        return ""
    if n_images == 1:
        return "Describe this image."
    return "Describe these images."


def _user_message_row(display_text: str, image_items: list[dict] | None) -> html.Div:
    bubble_children: list = []
    if display_text:
        bubble_children.append(html.Div(display_text, className="msg-text"))
    for item in image_items or []:
        src = _img_data_url(item)
        if src:
            bubble_children.append(html.Img(src=src, className="msg-img", alt="Attachment"))
    if not bubble_children:
        bubble_children.append(html.Div("(empty)", className="msg-text"))
    return html.Div(
        className="msg-row msg-user",
        children=[
            html.Div("You", className="msg-avatar"),
            html.Div(className="msg-bubble", children=bubble_children),
        ],
    )


def _assistant_message_row(inner) -> html.Div:
    return html.Div(
        className="msg-row msg-assistant",
        children=[
            html.Div("AI", className="msg-avatar"),
            html.Div(className="msg-bubble", children=[inner]),
        ],
    )


def _ensemble_details(replies: list[ProviderReply]) -> html.Details | None:
    if not replies:
        return None
    if len(replies) == 1 and replies[0].error is None:
        return None
    children: list = [html.Summary("Provider outputs")]
    for r in replies:
        body = r.error if r.error else (r.text or "")
        children.append(
            html.Div(
                className="sub-answer",
                children=[
                    html.H4(f"{r.label} ({r.provider_id})"),
                    html.Div(body),
                ],
            )
        )
    return html.Details(className="details-block", open=False, children=children)


def _api_key_form_rows() -> list:
    rows: list = []
    for pid, spec in sorted(PROVIDERS.items()):
        caption = provider_sidebar_model_caption(pid)
        rows.append(
            html.Div(
                className="owui-api-key-row",
                children=[
                    html.Label(
                        className="owui-api-key-label",
                        children=[
                            spec.label,
                            html.Span(f" ({spec.env_var})", className="owui-api-key-var"),
                        ],
                    ),
                    html.Div(caption, className="owui-api-key-model-hint"),
                    dcc.Input(
                        id={"type": "api-key", "env": spec.env_var},
                        type="password",
                        placeholder="Paste key…",
                        autoComplete="off",
                        className="owui-api-key-input",
                    ),
                ],
            )
        )
    return rows


def create_app() -> Dash:
    init_ensemble_opts = _ensemble_checklist_options(None)
    init_ensemble_val = _default_ensemble_value(init_ensemble_opts)

    app = Dash(
        __name__,
        assets_folder=str(_ROOT / "assets"),
        suppress_callback_exceptions=True,
        meta_tags=[
            {
                "http-equiv": "Cache-Control",
                "content": "no-cache, no-store, must-revalidate",
            },
            {"http-equiv": "Pragma", "content": "no-cache"},
            {"http-equiv": "Expires", "content": "0"},
        ],
    )

    app.layout = html.Div(
        className="owui-app",
        children=[
            dcc.Store(id="image-store", data=[]),
            dcc.Store(id="user-api-keys", storage_type="session", data=None),
            html.Aside(
                className="owui-sidebar",
                children=[
                    html.Div(
                        className="owui-brand",
                        children=[
                            html.Div("◆", className="owui-logo"),
                            html.Div(
                                children=[
                                    html.Span("Parallel Prompt", className="owui-title"),
                                    html.Span("Multi-GenAI chat", className="owui-tagline"),
                                ]
                            ),
                        ],
                    ),
                    html.Details(
                        className="owui-api-keys",
                        open=False,
                        children=[
                            html.Summary("API keys"),
                            html.P(
                                "Uses keys from your .env file, the process environment, or the fields below "
                                "(saved in this browser session only).",
                                className="owui-api-keys-hint",
                            ),
                            html.Div(
                                className="owui-api-key-fields",
                                children=_api_key_form_rows(),
                            ),
                            html.Button(
                                "Save keys",
                                id="save-api-keys",
                                n_clicks=0,
                                type="button",
                                className="owui-select-all",
                            ),
                        ],
                    ),
                    html.Div(
                        className="owui-provider-block",
                        children=[
                            html.Div(
                                className="owui-provider-toolbar",
                                children=[
                                    html.Div("", className="owui-section-label"),
                                    html.Button(
                                        "Select all",
                                        id="select-all-providers",
                                        n_clicks=0,
                                        className="owui-select-all",
                                        type="button",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="owui-checklist-scroll",
                                children=[
                                    dcc.Checklist(
                                        id="ensemble-providers",
                                        options=init_ensemble_opts,
                                        value=init_ensemble_val,
                                        labelStyle={"display": "block"},
                                        inputStyle={"marginRight": "0.45rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Main(
                className="owui-main",
                children=[
                    html.Header(
                        className="owui-topbar",
                        children=[html.Div("Chat", className="owui-topbar-title")],
                    ),
                    html.Div(
                        className="owui-chat-scroll",
                        children=[
                            dcc.Loading(
                                id="loading",
                                type="circle",
                                color="#3b82f6",
                                children=html.Div(
                                    id="answer-body",
                                    className="owui-messages",
                                    children=[_empty_state()],
                                ),
                            )
                        ],
                    ),
                    html.Div(
                        className="upload-preview-bar",
                        children=[
                            html.Div(
                                className="upload-preview-row",
                                children=[
                                    html.Div(
                                        id="upload-preview-thumb",
                                        className="upload-preview-thumb",
                                    ),
                                    html.Button(
                                        "×",
                                        id="remove-image",
                                        n_clicks=0,
                                        className="owui-remove-attachment",
                                        type="button",
                                        title="Remove images",
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="owui-composer",
                        children=[
                            html.Div(
                                className="owui-composer-inner",
                                children=[
                                    html.Div(
                                        className="owui-attach",
                                        children=[
                                            dcc.Upload(
                                                id="upload",
                                                children=html.Div("📎", title="Attach images"),
                                                className="upload-box",
                                                accept="image/png, image/jpeg, image/gif, image/webp",
                                                multiple=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="prompt-wrap",
                                        children=[
                                            dcc.Textarea(
                                                id="prompt",
                                                placeholder="Send a message…",
                                                value="",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="owui-send-wrap",
                                        children=[
                                            html.Button(
                                                "↑",
                                                id="submit",
                                                n_clicks=0,
                                                className="btn-send",
                                                title="Send",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    _hidden = {"display": "none"}
    _shown = {"display": "flex"}

    @callback(
        Output("image-store", "data"),
        Output("upload-preview-thumb", "children"),
        Output("remove-image", "style"),
        Input("upload", "contents"),
    )
    def stash_image(contents):
        if not contents:
            return [], None, _hidden
        parts = contents if isinstance(contents, list) else [contents]
        if len(parts) > MAX_ATTACHMENTS:
            return (
                [],
                html.Div(
                    f"Too many images at once (max {MAX_ATTACHMENTS}).",
                    className="err",
                ),
                _shown,
            )
        stored: list[dict] = []
        thumbs: list = []
        for c in parts:
            raw, mime = _parse_data_uri(c)
            if raw is None:
                return (
                    [],
                    html.Div("Could not read one or more images.", className="err"),
                    _shown,
                )
            if len(raw) > MAX_IMAGE_BYTES:
                return (
                    [],
                    html.Div("Each image must be 10 MB or smaller.", className="err"),
                    _shown,
                )
            stored.append(
                {"b64": base64.b64encode(raw).decode("ascii"), "mime": mime or "image/png"}
            )
            thumbs.append(html.Img(src=c, className="upload-preview-img", alt="Attached"))
        return (
            stored,
            html.Div(className="upload-preview-thumbs", children=thumbs),
            _shown,
        )

    @callback(
        Output("image-store", "data", allow_duplicate=True),
        Output("upload-preview-thumb", "children", allow_duplicate=True),
        Output("remove-image", "style", allow_duplicate=True),
        Input("remove-image", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_attached_image(_n_clicks):
        return [], None, _hidden

    @callback(
        Output("user-api-keys", "data"),
        Input("save-api-keys", "n_clicks"),
        State({"type": "api-key", "env": ALL}, "value"),
        State({"type": "api-key", "env": ALL}, "id"),
        State("user-api-keys", "data"),
        prevent_initial_call=True,
    )
    def save_user_api_keys(_n, values, id_list, prev):
        merged = dict(prev or {})
        if id_list and values is not None and len(values) == len(id_list):
            for raw, comp_id in zip(values, id_list):
                env_var = comp_id["env"]
                if raw and str(raw).strip():
                    merged[env_var] = str(raw).strip()
                else:
                    merged.pop(env_var, None)
        return merged

    @callback(
        Output("ensemble-providers", "options"),
        Output("ensemble-providers", "value", allow_duplicate=True),
        Input("user-api-keys", "data"),
        State("ensemble-providers", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_ensemble_from_keys(session_data, current_value):
        opts = _ensemble_checklist_options(
            session_data if isinstance(session_data, dict) else None
        )
        allowed = {o["value"] for o in opts if not o.get("disabled")}
        cur = [x for x in (current_value or []) if x in allowed]
        if not cur and allowed:
            cur = [sorted(allowed)[0]]
        return opts, cur

    @callback(
        Output("ensemble-providers", "value", allow_duplicate=True),
        Input("select-all-providers", "n_clicks"),
        State("ensemble-providers", "options"),
        prevent_initial_call=True,
    )
    def select_all_enabled_only(_n, options):
        return [o["value"] for o in (options or []) if not o.get("disabled")]

    @callback(
        Output("answer-body", "children"),
        Input("submit", "n_clicks"),
        State("prompt", "value"),
        State("ensemble-providers", "value"),
        State("image-store", "data"),
        State("user-api-keys", "data"),
        prevent_initial_call=True,
    )
    def run_prompt(n_clicks, prompt, ensemble_ids, image_data, user_keys):
        text = (prompt or "").strip()
        img_list = _decode_images_from_store(image_data)
        bubble_items = _store_image_items(image_data)
        if not text and not img_list:
            u = _user_message_row("", None)
            inner = html.Div(className="err", children="Enter a message and/or attach an image.")
            a = _assistant_message_row(inner)
            return html.Div([u, a])

        display_user = text if text else _default_vision_prompt(len(img_list))
        user_row = _user_message_row(display_user, bubble_items)

        session = user_keys if isinstance(user_keys, dict) else None
        file_vals = read_dotenv_file()
        raw_ids = list(ensemble_ids or [])
        ids = [
            pid
            for pid in raw_ids
            if _provider_key_configured(PROVIDERS[pid].env_var, session, file_vals)
        ]
        if not raw_ids:
            inner = html.Div(
                className="err",
                children="Select at least one provider in the sidebar.",
            )
            return html.Div([user_row, _assistant_message_row(inner)])
        if not ids:
            inner = html.Div(
                className="err",
                children="No API key for the selected provider(s). Add keys in “API keys” or your .env file.",
            )
            return html.Div([user_row, _assistant_message_row(inner)])

        with _api_env_session(session):
            vision_prompt = text or _default_vision_prompt(len(img_list))
            if len(ids) == 1:
                pid = ids[0]
                r = (
                    complete_safe_multimodal(pid, vision_prompt, images=img_list)
                    if img_list
                    else complete_safe_multimodal(pid, text)
                )
                if r.error:
                    inner = html.Div(className="err", children=r.error)
                else:
                    inner = dcc.Markdown(
                        _reply_to_markdown(r.text),
                        className="markdown",
                    )
                return html.Div([user_row, _assistant_message_row(inner)])

            replies, summary, err = (
                ensemble_core(ids, vision_prompt, images=img_list)
                if img_list
                else ensemble_core(ids, text)
            )
            parts: list = []
            if err:
                parts.append(html.Div(className="err", children=err))
            if summary:
                parts.append(
                    dcc.Markdown(
                        _reply_to_markdown(summary),
                        className="markdown",
                    )
                )
            det = _ensemble_details(replies)
            if det:
                parts.append(det)
            inner = html.Div(children=parts) if parts else html.Div("_Empty response._")
            return html.Div([user_row, _assistant_message_row(inner)])

    return app


def main() -> None:
    from parr_prompt.env_loader import load_env

    load_env()
    port = int(os.environ.get("PORT", "8050"))
    # Browsers cannot open http://0.0.0.0/ (invalid address). Default to loopback;
    # set HOST=0.0.0.0 to listen on all interfaces (Docker / some remote setups).
    host = os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1"
    app = create_app()
    # Default DEBUG=1 so the server hot-reloads on save; set DEBUG=0 to disable.
    _dbg = os.environ.get("DEBUG", "1").strip().lower()
    debug_on = _dbg not in ("0", "false", "no")
    open_url = f"http://127.0.0.1:{port}/"
    print(f"\n  Par Prompt — open this in your browser: {open_url}")
    if host == "0.0.0.0":
        print(
            "  (Server is bound to 0.0.0.0 — use 127.0.0.1 or localhost above, not 0.0.0.0.)\n"
        )
    else:
        print()
    app.run(debug=debug_on, host=host, port=port)


if __name__ == "__main__":
    main()
