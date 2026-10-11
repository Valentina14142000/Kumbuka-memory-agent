import time
import uuid
from functools import lru_cache

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from src import config, llm


class MemoryPayload(BaseModel):
    content: str = Field(..., description="One short, self-contained fact the user stated about themselves")


class MemoryExtractor(BaseModel):
    memories: list[MemoryPayload]


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    if config.QDRANT_PATH == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(path=config.QDRANT_PATH)


def _user_filter(user_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
    )


def ensure_collection() -> None:
    c = _client()
    if not c.collection_exists(config.COLLECTION_NAME):
        c.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=config.VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        try:
            c.create_payload_index(config.COLLECTION_NAME, field_name="user_id", field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass  # local mode does not need an index


def save_memory(user_id: str, text: str) -> bool:
    """Store a fact for this user. Returns False if it is empty or a near-duplicate."""
    text = text.strip()
    if not text:
        return False
    ensure_collection()
    vec = llm.embed([text], "RETRIEVAL_DOCUMENT")[0]
    similar = _client().query_points(
        collection_name=config.COLLECTION_NAME,
        query=vec,
        limit=1,
        query_filter=_user_filter(user_id),
        score_threshold=config.DUPLICATE_THRESHOLD,
    ).points
    if similar:
        return False
    _client().upsert(
        collection_name=config.COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"user_id": user_id, "content": text, "created_at": time.time()},
            )
        ],
    )
    return True


def retrieve_memories(user_id: str, query: str, limit: int = 3) -> list[str]:
    if not query.strip():
        return []
    ensure_collection()
    vec = llm.embed([query], "RETRIEVAL_QUERY")[0]
    hits = _client().query_points(
        collection_name=config.COLLECTION_NAME,
        query=vec,
        limit=limit,
        query_filter=_user_filter(user_id),
        score_threshold=config.RECALL_THRESHOLD,
    ).points
    return [h.payload["content"] for h in hits]


def list_memories(user_id: str) -> list[dict]:
    ensure_collection()
    points, _ = _client().scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=_user_filter(user_id),
        limit=500,
        with_payload=True,
    )
    rows = [
        {"id": str(p.id), "content": p.payload["content"], "created_at": p.payload.get("created_at", 0)}
        for p in points
    ]
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


def delete_memory(user_id: str, memory_id: str) -> bool:
    ensure_collection()
    found = _client().retrieve(collection_name=config.COLLECTION_NAME, ids=[memory_id], with_payload=True)
    if not found or found[0].payload.get("user_id") != user_id:
        return False
    _client().delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=models.PointIdsList(points=[memory_id]),
    )
    return True
