from pydantic import BaseModel, Field


class MemoryPayload(BaseModel):
    content: str = Field(...)


class MemoryExtractor(BaseModel):
    memories: list[MemoryPayload]


MEMORY_COLLECTION: list[MemoryPayload] = []


def ensure_collection():
    return MEMORY_COLLECTION


def retrieve_memories(query: str, limit: int = 3) -> list[str]:
    if not query:
        return []

    normalized = query.lower()
    matches = [memory.content for memory in MEMORY_COLLECTION if normalized in memory.content.lower()]
    return matches[:limit]


def save_memory_to_vector_db(memory: MemoryPayload | str) -> None:
    if isinstance(memory, str):
        memory = MemoryPayload(content=memory)

    if memory.content and memory not in MEMORY_COLLECTION:
        MEMORY_COLLECTION.append(memory)
