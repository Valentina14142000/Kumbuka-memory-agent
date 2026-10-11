import os

from src.graph import chat
from src.memory import ensure_collection, list_memories


def main() -> None:
    user_id = os.getenv("MEMORY_USER_ID", "default-user")
    ensure_collection()
    print(f"Memory Agent ready (user: {user_id}). Commands: /memories, exit\n")
    while True:
        try:
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if text == "/memories":
            rows = list_memories(user_id)
            print("\n".join(f"  - {r['content']}" for r in rows) if rows else "  (no memories yet)")
            continue
        try:
            out = chat(user_id, text)
        except RuntimeError as e:
            print(f"\n[error] {e}\n")
            continue
        if out["memories_used"]:
            print(f"  (recalled {len(out['memories_used'])} memories)")
        for m in out["memories_saved"]:
            print(f"  [saved] {m}")
        print(f"\nAgent: {out['reply']}\n")


if __name__ == "__main__":
    main()
