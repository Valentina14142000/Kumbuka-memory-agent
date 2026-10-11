import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import memory
from src.graph import chat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kumbuka")

ID = r"^[A-Za-z0-9_-]+$"
RATE = int(os.getenv("CHAT_PER_HOUR", "30"))
WEB = Path(__file__).resolve().parent.parent / "web"
_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="Kumbuka")


class ChatIn(BaseModel):
    user_id: str = Field(min_length=3, max_length=64, pattern=ID)
    message: str = Field(min_length=1, max_length=2000)


def _limit(request: Request) -> None:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, f"Limit reached: {RATE} messages per hour. Please try again later.")
    q.append(now)


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat_endpoint(body: ChatIn, request: Request):
    _limit(request)
    try:
        return chat(body.user_id, body.message)
    except RuntimeError as e:
        log.error("Chat failed: %s", e)
        raise HTTPException(502, str(e))
    except Exception:
        log.exception("Unexpected chat error")
        raise HTTPException(500, "Unexpected error while chatting. Check the server logs.")


@app.get("/memories")
def memories(user_id: str = Query(..., min_length=3, max_length=64, pattern=ID)):
    try:
        return memory.list_memories(user_id)
    except Exception as e:
        log.exception("Listing memories failed")
        raise HTTPException(503, f"Memory store error: {type(e).__name__}: {e}")


@app.delete("/memories/{memory_id}")
def forget(memory_id: str, user_id: str = Query(..., min_length=3, max_length=64, pattern=ID)):
    try:
        deleted = memory.delete_memory(user_id, memory_id)
    except Exception as e:
        log.exception("Deleting memory failed")
        raise HTTPException(503, f"Memory store error: {type(e).__name__}: {e}")
    if not deleted:
        raise HTTPException(404, "Memory not found")
    return {"deleted": memory_id}
