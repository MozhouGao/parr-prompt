# parr-prompt

Send prompts to several generative AI backends from the command line, optionally **fan out** the same prompt to multiple providers in parallel and **summarize** the successful replies with **Moonshot Kimi**. A small **Dash** web UI offers a chat-style layout with optional **image** uploads.

Supported providers: **OpenAI**, **Google Gemini**, **xAI Grok**, **Anthropic Claude**, **DeepSeek**, **Moonshot Kimi**, and **Doubao** (Volcengine Ark).

Requires **Python 3.10+**.

## Dependencies

Declared in **`pyproject.toml`** and mirrored in **`requirements.txt`**:

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI + OpenAI-compatible APIs (xAI Grok, DeepSeek, Kimi, Doubao) |
| `anthropic` | Anthropic Claude |
| `google-genai` | Google Gemini |
| `dash` | **`parr-web`** UI |
| `python-dotenv` | Load **`.env`** for CLI and web app |

## Install

From the directory that contains **`pyproject.toml`** (the project folder; this may be the repo root or the inner `parr_prompt/` folder depending on how you cloned). That installs dependencies from `pyproject.toml` and the `parr-prompt` / `parr-web` commands:

```bash
pip install -e .
```

Alternatively you can install the same dependency set from **`requirements.txt`**, then install the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

This installs two console scripts: `parr-prompt` and `parr-web`.

## API keys

Set environment variables for whichever providers you use. At minimum, export the keys you need for your workflow.

| Provider   | Environment variable   | Notes |
|-----------|-------------------------|--------|
| OpenAI    | `OPENAI_API_KEY`        | Default **`gpt-5.4-mini`** ([model docs](https://platform.openai.com/docs/models/gpt-5.4-mini)). Optional **`OPENAI_DEFAULT_MODEL`**. CLI: `-m`. |
| Gemini    | `GOOGLE_API_KEY`        | Default **`gemini-2.5-flash`**. Gemini only (not used for ensemble summary). CLI: `-m`. |
| Kimi      | `MOONSHOT_API_KEY`      | Used for ensemble **summary** when **2+** providers succeed |
| Kimi host | `MOONSHOT_BASE_URL`     | Optional. Default is `https://api.moonshot.ai/v1` (international keys). China-console keys: `https://api.moonshot.cn/v1` |
| Kimi model | `MOONSHOT_DEFAULT_MODEL` | Optional override for all Kimi calls (text, images, ensemble summary). Default in code: **`kimi-k2.5`**. Legacy alias: **`MOONSHOT_VISION_MODEL`** (used only if `MOONSHOT_DEFAULT_MODEL` is unset). |
| Grok      | `XAI_API_KEY`           | Text default **`grok-4-1-fast-non-reasoning`**; with an **image**, default **`grok-4-1-fast-reasoning`** (override with **`XAI_IMAGE_MODEL`**). **`XAI_DEFAULT_MODEL`** overrides the text default. **403** → enable those models for your key in **[console.x.ai](https://console.x.ai)**. CLI: `-m`. |
| Claude    | `ANTHROPIC_API_KEY`     | Default **`claude-sonnet-4-20250514`**. CLI: `-m`. |
| DeepSeek  | `DEEPSEEK_API_KEY`      | Default **`deepseek-chat`** (DeepSeek-V3.2 non-thinking); **`deepseek-reasoner`** = thinking. The [chat API](https://api-docs.deepseek.com/) accepts **text-only** `messages` (no **`image_url`** parts), so this app **skips** DeepSeek when a picture is attached. Optional **`DEEPSEEK_DEFAULT_MODEL`**. |
| Doubao    | `ARK_API_KEY`           | No default model; use `-m` with your Ark endpoint/model id |

A template with these variables (and optional `PORT` / `DEBUG` for the web UI) is in **`.env.example`**. Copy it to **`.env`**, fill in secrets, and keep **`.env` out of version control** (it is listed in `.gitignore`).

**`parr-prompt` and `parr-web`** load **`.env`** automatically from the **repository root** (parent of the `parr_prompt` package) and then from the **current working directory**, via **`python-dotenv`** (installed with the package). You can still export variables in your shell or use [direnv](https://direnv.net/) if you prefer.

List defaults and env var names:

```bash
parr-prompt --list
```

## Command line: single provider

```bash
parr-prompt -p openai "Explain attention in transformers in two sentences."
parr-prompt -p claude -f notes.txt
echo "Hello" | parr-prompt -p deepseek
```

Override the default model (single-provider mode only):

```bash
parr-prompt -p openai -m gpt-5.4 "Override the default gpt-5.4-mini when you need the full GPT-5.4 model."
```

Optional: `--max-tokens N` caps output length for that call.

## Command line: multiple providers + Kimi summary

The same prompt is sent to each listed provider **in parallel**. If **exactly one** provider returns a successful non-empty answer, that text is printed **directly** (no Kimi call). If **two or more** succeed, those answers are sent to **Moonshot (Kimi)** to produce a **Markdown table**: column **Provider** (label + id) and column **Short summary** (one concise summary per GenAI). The **original user question is not** included in that synthesis prompt.

`MOONSHOT_API_KEY` is required **only** when at least two providers succeed (Kimi summary step). If you list a single provider, or only one succeeds, Kimi is not used.

If Kimi returns **401 Invalid Authentication**, your key and API host may not match: keys from [platform.moonshot.ai](https://platform.moonshot.ai) use the default `.ai` endpoint; keys from the China console need **`MOONSHOT_BASE_URL=https://api.moonshot.cn/v1`** in `.env`.

If the error text tells you to fix your key at **platform.openai.com**, the request went to **OpenAI**, not Moonshot. That usually means **OpenAI** is the selected provider while the key in **`OPENAI_API_KEY`** is actually a Moonshot key (or another non-OpenAI key). Select **Moonshot Kimi** and put the key under **`MOONSHOT_API_KEY`**, or reinstall from this repo (`pip install -e .`) so an older build cannot send Kimi to OpenAI with a missing base URL.

```bash
parr-prompt --providers openai,claude,grok "What is RAG?"
```

Useful options:

- `--max-tokens N` — per-provider cap on fan-out calls  
- `--synth-model ID` — Kimi model for the summary (default: `kimi-k2.5`)  
- `--synth-max-tokens N` — cap on the summary length  
- `-v` / `--verbose` — print each provider’s reply on **stderr** before printing the summary on **stdout**

Skipped providers (missing key, unsupported case, or API error) are reported on stderr; the run continues if at least one provider succeeds.

## Web UI

```bash
parr-web
```

Open **http://127.0.0.1:8050** (or **http://localhost:8050**). Do **not** use `http://0.0.0.0:8050` in the browser — that address is invalid for visiting a page (it is only for binding the server).

Override the port with **`PORT`**. By default the server listens on **`127.0.0.1`**; set **`HOST=0.0.0.0`** if you need every interface (e.g. some containers); you should still open **127.0.0.1** or **localhost** locally.

The interface is laid out like **Open WebUI**: a **left sidebar** (**API keys** form + provider checklist + **Select all**), a **main chat** area, and a **bottom composer**.

- **Default model ids** — Under each provider name in the **GenAIs** checklist (and under each row in **API keys**), the UI shows the **effective default model** (honoring **`OPENAI_DEFAULT_MODEL`**, **`XAI_*`**, **`MOONSHOT_*`**, **`DEEPSEEK_DEFAULT_MODEL`**, etc.). **Friendly UI labels** (API ids unchanged): **`deepseek-chat`** → **DeepSeek-v3.2**, **`deepseek-reasoner`** → **DeepSeek-v3.2 (reasoner)**; Claude **`claude-sonnet-4-*`** snapshot ids → **`claude-sonnet-4`**.
- **Provider checkboxes** are **disabled** until a key exists for that provider: non-empty value in the repo **`.env`** file (parsed on each sync), **`os.environ`** (including values loaded at startup by `python-dotenv`), or keys you paste under **API keys** and **Save** (stored in the **browser session** only).
- **One provider checked** — single-model reply.  
- **Two or more checked** — parallel calls plus **Kimi** summary when multiple succeed (expandable **Provider outputs** when useful).  
- **Images (optional)** — attach **one or many** PNG, JPEG, GIF, or WebP files in a single picker (up to **12** images, **10 MB** each). Image+text works for **OpenAI, Gemini, Claude, Grok, and Kimi**. **DeepSeek** and **Doubao** are skipped (DeepSeek’s API rejects multimodal message parts; only plain text is allowed in `content`).

Set the same API keys in the environment as for the CLI. `MOONSHOT_API_KEY` is needed for ensemble mode **only when two or more providers return a successful answer** (so Kimi can summarize). A single successful reply does not call Kimi.

`parr-web` runs with Dash **debug / hot reload on by default** (`DEBUG=1`). Set `DEBUG=0` to turn it off.

**If the sidebar still shows an old layout** (e.g. “PROVIDER”, “Multi + summary”, or the keys footer), you are not running the current code: stop the server, run `pip install -e .` from **this** repository root, start `parr-web` again, and hard-refresh the browser (Ctrl+Shift+R). If you have two copies of the project (for example Windows and WSL paths), make sure both the install and the server use the same folder.

## Vision / images

Multimodal calls use each vendor’s image APIs where implemented. If you attach image(s) but no text, the app uses a short default such as “Describe this image.” or “Describe these images.” when multiple files are attached.

- **Kimi** — Default model **`kimi-k2.5`** for both text and images ([Moonshot multimodal guide](https://platform.moonshot.ai/docs/guide/use-kimi-vision-model)). Override with **`MOONSHOT_DEFAULT_MODEL`** or `-m`. If you still have **`MOONSHOT_VISION_MODEL=moonshot-v1-8k`** in `.env`, remove it or point it at **`kimi-k2.5`** — that old variable was forcing a non-vision model and caused “image not supported” errors.
- **DeepSeek** — **V3.2** text only: **`deepseek-chat`** / **`deepseek-reasoner`** via **`-m`** or **`DEEPSEEK_DEFAULT_MODEL`**. The API only allows **text** in message `content`; sending **`image_url`** parts triggers errors like `unknown variant image_url, expected text`. For image+text, use Kimi, OpenAI, Gemini, Claude, or Grok.
- **Doubao** — No image path in this project; use a vision-capable provider or text-only prompts.

## Development

The package layout is `parr_prompt/` (CLI in `cli.py`, providers in `providers.py`, ensemble logic in `ensemble.py`, Dash app in `web_app.py`). Run the CLI module directly if you prefer:

```bash
python -m parr_prompt --list
```

That prints each provider’s env var, **default model id** (from code, before your **`-m`** override), and any **base URL** for OpenAI-compatible backends.
