# Kumbuka

**An AI chat that remembers you.** *Kumbuka* means "remember" in Swahili.

**Live demo:** https://YOUR-RENDER-ADDRESS
(Free hosting: the first load can take up to a minute while the server wakes up. Please don't share sensitive information.)

Tell Kumbuka about your tools, preferences, and projects. It extracts the lasting facts, stores them as vectors, and recalls the relevant ones in later conversations, even after you close the page.

## How it works

1. **Recall:** your newest message is embedded and matched against your stored memories in Qdrant.
2. **Answer:** Gemini replies using the recalled facts and your last 12 messages (short-term memory, kept by a LangGraph checkpointer).
3. **Remember:** your message is scanned for lasting facts you stated about yourself. New, non-duplicate facts are embedded and stored.

One-off requests ("how do I bake a cake?") are not meant to be saved. Memories are separated per visitor by an anonymous ID stored in the browser.

## Tech stack

Python, FastAPI, LangGraph, Google Gemini (`google-genai`), Qdrant (local mode or Qdrant Cloud), vanilla HTML/CSS/JS front end.

## Run locally

```bash
git clone https://github.com/Valentina14142000/kumbuka.git
cd kumbuka
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key" > .env
python -m pytest -v                       # tests need no API key
python -m uvicorn src.api:app --port 8000 # web app at http://127.0.0.1:8000
python app.py                             # or: terminal chat
```

Get a free key at [Google AI Studio](https://aistudio.google.com). Local mode stores memories in `qdrant_data/` and needs no Docker. Only one process can use that folder at a time, so stop the terminal chat before starting the web app.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini key (required) | none |
| `GEMINI_MODELS` | Chat models, tried in order | `gemini-3.8-flash,gemini-3.5-flash` |
| `EMBED_MODEL`, `VECTOR_SIZE` | Embedding model and its dimension | `gemini-embedding-001`, `768` |
| `QDRANT_URL`, `QDRANT_API_KEY` | Use Qdrant Cloud instead of local files | unset |
| `RECALL_THRESHOLD` | Minimum similarity to recall a memory | `0.55` |
| `CHAT_PER_HOUR` | Messages per IP per hour | `30` |

Model names change over time. If you see a 404, set `GEMINI_MODELS` or `EMBED_MODEL` to names your key supports.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | `{user_id, message}` returns the reply, memories used, memories saved |
| `GET` | `/memories?user_id=` | List a visitor's memories |
| `DELETE` | `/memories/{id}?user_id=` | Delete one memory |
| `GET` | `/health` | Liveness check |

## Deploy (Render + Qdrant Cloud)

1. Create a free cluster at cloud.qdrant.io and copy its URL and an API key.
2. On Render, create a Web Service from this repo. Build: `pip install -r requirements.txt`. Start: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`. Health check: `/health`.
3. Set `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `PYTHON_VERSION=3.13.3`.

The web app and API are served by the same service, so no CORS setup is needed.

## Limitations and privacy

- Messages are sent to Google Gemini. The free tier may use submitted content to improve Google's products, so use non-sensitive information only.
- Visitors are identified only by a random ID in their browser. Anyone who has that ID can read those memories, and clearing browser data loses access.
- Short-term chat history lives in server memory and resets on restart. Long-term memories persist in Qdrant.
- Memories come from a model's judgment and can occasionally be wrong or miss facts.
- Free tiers (Render, Qdrant Cloud, Gemini) have limits and cold starts, and all visitors share one Gemini quota.

## Roadmap

- [x] Vector memory with per-visitor separation and duplicate detection
- [x] Retries and model fallback
- [x] Web UI and tests
- [ ] "Forget everything" control in the UI
- [ ] Persistent chat history
- [ ] Evaluation set for memory extraction quality

## License

MIT
