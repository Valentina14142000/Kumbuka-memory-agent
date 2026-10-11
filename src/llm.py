import time

from google.genai import errors, types

from src import config

BUSY = {429, 500, 502, 503, 504}


def generate(contents, gen_config=None, retries: int = 3) -> str:
    """Call Gemini with retries on busy errors, then fall back to the next model."""
    last = "no response"
    for model in config.CHAT_MODELS:
        for attempt in range(retries):
            try:
                res = config.get_client().models.generate_content(
                    model=model, contents=contents, config=gen_config
                )
                return res.text or ""
            except errors.APIError as e:
                last = f"{model}: {e.code} {e.message}"
                if e.code in BUSY:
                    time.sleep(2**attempt)
                    continue
                break  # 404 or auth error: try the next model
    raise RuntimeError(f"All models failed. Last error: {last}")


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    last = "no response"
    for attempt in range(3):
        try:
            res = config.get_client().models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=config.VECTOR_SIZE
                ),
            )
            return [list(e.values) for e in res.embeddings]
        except errors.APIError as e:
            last = f"{e.code} {e.message}"
            if e.code in BUSY:
                time.sleep(2**attempt)
                continue
            break
    raise RuntimeError(f"Embedding failed ({config.EMBED_MODEL}): {last}")
