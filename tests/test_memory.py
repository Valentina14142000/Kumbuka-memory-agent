import hashlib
import re

import pytest
from fastapi.testclient import TestClient

from src import config, llm, memory


def fake_embed(texts, task_type):
    out = []
    for t in texts:
        v = [0.0] * config.VECTOR_SIZE
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % config.VECTOR_SIZE] += 1.0
        out.append(v)
    return out


@pytest.fixture(autouse=True)
def store(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_PATH", ":memory:")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "RECALL_THRESHOLD", 0.1)
    monkeypatch.setattr(llm, "embed", fake_embed)
    memory._client.cache_clear()
    yield
    memory._client.cache_clear()


def test_save_and_recall():
    assert memory.save_memory("alice", "I build services with FastAPI and Qdrant")
    hits = memory.retrieve_memories("alice", "how should I structure a FastAPI service")
    assert hits == ["I build services with FastAPI and Qdrant"]


def test_duplicates_are_not_stored_twice():
    assert memory.save_memory("alice", "I prefer Qdrant over Pinecone")
    assert not memory.save_memory("alice", "I prefer Qdrant over Pinecone")
    assert len(memory.list_memories("alice")) == 1


def test_unrelated_query_returns_nothing():
    memory.save_memory("alice", "I build services with FastAPI")
    assert memory.retrieve_memories("alice", "recipe for pancakes") == []


def test_users_are_isolated():
    memory.save_memory("alice", "I build services with FastAPI")
    assert memory.retrieve_memories("bob", "FastAPI services") == []
    assert memory.list_memories("bob") == []


def test_cannot_delete_another_users_memory():
    memory.save_memory("alice", "I use LangGraph")
    mid = memory.list_memories("alice")[0]["id"]
    assert not memory.delete_memory("bob", mid)
    assert memory.delete_memory("alice", mid)
    assert memory.list_memories("alice") == []


def test_api_validation():
    from src.api import app

    c = TestClient(app)
    assert c.get("/health").json() == {"ok": True}
    assert c.get("/memories").status_code == 422
    assert c.post("/chat", json={"user_id": "x", "message": "hi"}).status_code == 422
