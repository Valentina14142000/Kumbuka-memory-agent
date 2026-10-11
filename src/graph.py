import logging
from typing import Annotated, Sequence

from google.genai import types
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src import config, llm
from src.memory import MemoryExtractor, retrieve_memories, save_memory

log = logging.getLogger("memory-agent")


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    retrieved_memories: list[str]
    saved_memories: list[str]


def _last_human(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return str(m.content)
    return ""


def retrieve_node(state: AgentState):
    """Look up long-term memories relevant to the newest user message."""
    try:
        return {"retrieved_memories": retrieve_memories(state["user_id"], _last_human(state), limit=3)}
    except Exception as e:  # memory trouble must not break the chat
        log.warning("Memory recall failed: %s", e)
        return {"retrieved_memories": []}


def agent_node(state: AgentState):
    """Answer using recalled memories plus a trimmed window of recent messages."""
    context = "\n".join(f"- {m}" for m in state.get("retrieved_memories", []))
    system_prompt = f"""You are a direct, helpful AI collaborator for a developer.

What you remember about the user from earlier conversations:
{context if context else "Nothing relevant yet."}

Use these facts naturally when they help. Never invent facts about the user.
Answer concisely, as a colleague would."""

    window = list(state["messages"])[-config.HISTORY_WINDOW:]
    while window and window[0].type != "human":
        window.pop(0)
    contents = [
        types.Content(
            role="user" if m.type == "human" else "model",
            parts=[types.Part.from_text(text=str(m.content))],
        )
        for m in window
    ]
    text = llm.generate(
        contents, types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7)
    )
    return {"messages": [AIMessage(content=text)]}


def memory_node(state: AgentState):
    """Extract lasting facts the USER stated about themselves and store them."""
    user_text = _last_human(state)
    if not user_text:
        return {"saved_memories": []}
    prompt = (
        "From the user's message below, extract lasting facts they stated about themselves: "
        "preferences, tech stack, projects, constraints. Write each as one short self-contained "
        "sentence. Ignore questions, small talk, and anything the user did not say. "
        "Return an empty list if there is nothing worth remembering.\n\n"
        f"User message: {user_text}"
    )
    try:
        raw = llm.generate(
            prompt,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryExtractor,
                temperature=0.0,
            ),
        )
        facts = MemoryExtractor.model_validate_json(raw).memories
        saved = [f.content for f in facts if save_memory(state["user_id"], f.content)]
        return {"saved_memories": saved}
    except Exception as e:
        log.warning("Memory extraction failed: %s", e)
        return {"saved_memories": []}


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("agent", agent_node)
workflow.add_node("memory", memory_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", "memory")
workflow.add_edge("memory", END)

# Short-term memory: the conversation for each user_id lives in this checkpointer.
compiled_graph = workflow.compile(checkpointer=MemorySaver())


def chat(user_id: str, message: str) -> dict:
    out = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "retrieved_memories": [],
            "saved_memories": [],
        },
        config={"configurable": {"thread_id": user_id}},
    )
    return {
        "reply": str(out["messages"][-1].content),
        "memories_used": out.get("retrieved_memories", []),
        "memories_saved": out.get("saved_memories", []),
    }
