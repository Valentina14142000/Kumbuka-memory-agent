from fastapi.testclient import TestClient

from src import llm, suggest


def test_suggestions_are_cleaned(monkeypatch):
    raw = '{"suggestions": ["  How do I add tests?  ", "", "x", "Show me an example", "What about auth?", "Another one"]}'
    monkeypatch.setattr(llm, "generate", lambda *a, **k: raw)
    out = suggest.next_prompts("I use FastAPI", "Nice choice.")
    assert out == ["How do I add tests?", "Show me an example", "What about auth?"]


def test_suggest_endpoint_validates_and_never_fails(monkeypatch):
    from src import api

    c = TestClient(api.app)
    assert c.post("/suggest", json={"user_id": "x", "message": "a", "reply": "b"}).status_code == 422

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(api, "next_prompts", boom)
    r = c.post("/suggest", json={"user_id": "tester-123", "message": "hi", "reply": "hello"})
    assert r.status_code == 200 and r.json() == {"suggestions": []}
