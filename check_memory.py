import traceback

from src import config, llm, memory

print("chat models:", config.CHAT_MODELS)
print("embed model:", config.EMBED_MODEL, "dim", config.VECTOR_SIZE)
print("qdrant:", config.QDRANT_URL or config.QDRANT_PATH)

try:
    v = llm.embed(["hello"], "RETRIEVAL_DOCUMENT")
    print("embedding OK, dimension", len(v[0]))
except Exception as e:
    print("EMBEDDING ERROR:", e)

try:
    memory.ensure_collection()
    print("qdrant collection OK")
except Exception:
    print("QDRANT ERROR:")
    traceback.print_exc()

try:
    print("saved:", memory.save_memory("diag-user-1", "I build APIs with FastAPI"))
    print("list:", memory.list_memories("diag-user-1"))
    print("recall:", memory.retrieve_memories("diag-user-1", "What framework do I use?"))
except Exception:
    traceback.print_exc()
